#!/usr/bin/env python3
"""Fail-closed qualifier for the bounded M265 Weyl-loop calibration."""

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
PRODUCTION = PACKAGE / "quantum_weyl_loop_bus.py"
REFERENCE = PACKAGE / "tests" / "quantum_weyl_loop_bus_separate_reference.py"
CONTRACT = PACKAGE / "PHASE_QEMU_V7_WEYL_LOOP_CONTRACT.md"
FINDINGS = PACKAGE / "PHASE_QEMU_V7_WEYL_LOOP_FINDINGS.md"
PRODUCTION_SEAL = PACKAGE / "evidence" / "PHASE_QEMU_V7_WEYL_LOOP_BUS.json"
REFERENCE_SEAL = (
    PACKAGE / "evidence" / "PHASE_QEMU_V7_WEYL_LOOP_BUS_SEPARATE_REFERENCE.json"
)

CLAIM = (
    "IDEAL_INFINITE_CCR_WEYL_COMMUTATOR_FACTORIZATION_WITH_ARBITRARY_NORMAL_"
    "STATE_BUS_IDENTITY_AND_FINITE_ENERGY_CONSTRAINED_TRUNCATED_FOCK_NUMERICAL_"
    "CONVERGENCE_ON_A_BOUNDED_THREE_QUBIT_CALIBRATION_LOAD"
)
CEILING = (
    "DETERMINISTIC_COMPLEX128_SOFTWARE_EMULATION_WITH_LOGICAL_RESIDENT_ARRAY_"
    "CUSTODY_DIRECT_COMPILED_FORWARD_SHADOW_AND_NO_PHYSICAL_SAME_MODE_CUSTODY"
)
RESTORATION = "NUMERICAL_PHYSICAL_STATE_RESTORATION"
RESTORATION_SCOPE = (
    "ENERGY_CONSTRAINED_COMPLEX128_LOGICAL_RESIDENT_BACKING_BUS_AND_REFERENCE_"
    "RETURN_WITH_CLIENT_TRANSFORMATION_AND_COMPLETE_FACTORIZATION_AT_CUTOFF128_"
    "WITHOUT_PHYSICAL_SAME_MODE_CUSTODY"
)
DISPOSITION = (
    "DIRECT_COMPILED_ZZ_FORWARD_SHADOW_STRICTLY_OMITS_THE_BUS_LOOP_AND_NO_"
    "RESOURCE_ADVANTAGE_OR_M257_ESCAPE_IS_ESTABLISHED"
)
NEXT = (
    "MULTIMODE_TRAPPED_ION_STATE_DEPENDENT_FORCE_WEYL_LOOP_DIGITAL_TWIN_WITH_"
    "HEATING_SPECTATOR_MODE_CLOSURE_CONTROLLER_COST_AND_ENERGY_CONSTRAINED_"
    "SAME_MODE_REUSE"
)
REFERENCE_CEILING = (
    "FINITE_COMPLEX128_DETERMINISTIC_SOFTWARE_REFERENCE_WITH_TRUNCATED_FOCK_"
    "CCR_AND_DIRECT_CLIENT_ORACLE_NO_PHYSICAL_OR_SAME_BACKING_RESTORATION"
)
REFERENCE_DISPOSITION = (
    "IDEAL_WEYL_LOOP_IS_A_CATALYTIC_INTERACTION_LAW_BUT_FINITE_CUTOFF_RETURN_"
    "IS_ONLY_ENERGY_CONSTRAINED_AND_THE_EXACT_DIRECT_COMPILED_CLIENT_SHADOW_"
    "RETAINS_M257"
)

EXPECTED_HASHES = {
    PRODUCTION: "43c7c140538bdbf0eec2a7906bd9a2b81b0ee28e6b9f2de00a587ba9ecb853f8",
    REFERENCE: "844a5040e7f8402fb6d872a951cf1ea2cece1dd666a39a0b55622f3703f88a4e",
    CONTRACT: "3c393632acf9c25fbe2bdf74352714604024163b0d954c513ecedd7159f2affc",
    FINDINGS: "29babfeac5eb03b93d27d6a902ce76740629c29500a1732b252fab1a9114583a",
}

CUTOFFS = (16, 32, 64, 128)
PRODUCTION_FIXTURES = {
    "vacuum",
    "coherent_alpha_0p65_plus_0p20i",
    "thermal_nbar_0p4",
    "squeezed_r_0p45_phi_0p30",
    "fock_1_non_gaussian",
    "fock_superposition_0_plus_i3",
    "phi4_bus_reference",
}
REFERENCE_FIXTURES = {
    "vacuum",
    "coherent_0p65_plus_0p20i",
    "thermal_nbar_0p4",
    "squeezed_r0p45_phi0p30",
    "fock_1",
    "zero_plus_i_three",
    "phi4_bus_reference",
}
FINAL_BOUNDARY = {
    "X0": 1.0 / math.sqrt(2.0),
    "X1": 1.0 / (2.0 * math.sqrt(2.0)),
    "X2": 0.5,
    "Y0Z1": 1.0 / math.sqrt(2.0),
    "Z0Y1": 1.0 / (2.0 * math.sqrt(2.0)),
    "Z1Y2": math.sqrt(3.0) / 2.0,
    "X0X1": 0.5,
    "X1X2": 1.0 / math.sqrt(2.0),
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
    """Normalize only fields explicitly declared nondeterministic/nonclaim."""

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


def fixture_map(production: Mapping[str, Any], cutoff: int) -> dict[str, Any]:
    records = production["cutoff_sweep"][str(cutoff)]
    return {record["fixture"]: record for record in records}


def source_and_document_audit() -> None:
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"missing dependency: {path.name}")
        require(sha256(path) == expected, f"dependency changed: {path.name}")

    reference_source = REFERENCE.read_text(encoding="utf-8")
    require(PRODUCTION.name not in reference_source, "reference names production")
    require("import quantum_weyl_loop_bus" not in reference_source, "reference imports production")

    production_source = PRODUCTION.read_text(encoding="utf-8")
    for source_anchor in (
        "diagnostics_a = run_schedule_in_place(state, cache, accepted_schedule(PROGRAM_A))",
        "diagnostics_b = run_schedule_in_place(state, cache, accepted_schedule(PROGRAM_B))",
        '"allocation_object_unchanged": id(state) == backing_object_id',
        '"allocation_base_pointer_unchanged": int(state.__array_interface__["data"][0])',
        '"guest_boundary_release_count": 1',
        '"snapshot_count": 0',
        '"reload_count": 0',
    ):
        require(source_anchor in production_source, f"production order/custody anchor missing: {source_anchor}")

    for document in (CONTRACT, FINDINGS):
        text = document.read_text(encoding="utf-8")
        for value, label in (
            (CLAIM, "claim"),
            (CEILING, "ceiling"),
            (RESTORATION, "restoration class"),
            (RESTORATION_SCOPE, "restoration scope"),
            (DISPOSITION, "resource disposition"),
            (NEXT, "successor"),
        ):
            require(value in text, f"{label} missing from {document.name}")
        lowered = text.lower()
        for required in (
            "m257",
            "direct compiled",
            "top-fock",
            "physical same-mode",
            "energy-constrained",
            "inert reference",
        ):
            require(required in lowered, f"{required} scope missing from {document.name}")

    findings = FINDINGS.read_text(encoding="utf-8")
    require("positive restoration-law calibration and a negative computational-" in findings, "two-sided disposition missing")
    require("2.336739139525e-15" in findings, "joint factorization anchor missing")
    require("9.699833445253e-16" in findings, "bus return anchor missing")
    require("not a phase-native client machine" in findings, "client ceiling missing")


def metadata_audit(production: Mapping[str, Any], reference: Mapping[str, Any]) -> None:
    require(production["schema"] == "PHASE_QEMU_V7_WEYL_LOOP_BUS_RESULT_V1", "production schema changed")
    require(production["milestone"] == "M265", "milestone changed")
    require(production["status"] == "PASS_ENERGY_CONSTRAINED_LOGICAL_RETURN", "production status failed")
    require(production["source_self_assertion"] == production["status"], "source self assertion differs")
    require(production["source_sha256"] == EXPECTED_HASHES[PRODUCTION], "embedded production hash differs")
    require(production["contract_sha256"] == EXPECTED_HASHES[CONTRACT], "embedded contract hash differs")
    require(production["claim"] == CLAIM, "claim changed")
    require(production["claim_ceiling"] == CEILING, "ceiling changed")
    require(production["restoration_classification"] == RESTORATION, "restoration class changed")
    require(production["restoration_scope"] == RESTORATION_SCOPE, "restoration scope changed")
    require(production["resource_disposition"] == DISPOSITION, "resource disposition changed")
    require(production["next_mechanism"] == NEXT, "successor changed")
    require(production["registry_taxonomy_label_does_not_establish_physical_hardware"], "taxonomy caveat missing")
    require(not production["terminal"], "long-term goal was terminated")

    require(reference["reference_id"] == "M265_QUANTUM_WEYL_LOOP_BUS_SEPARATE_REFERENCE_V1", "reference id changed")
    require(reference["status"] == "PASS_SEPARATE_REFERENCE", "reference failed")
    require(reference["source_sha256"] == EXPECTED_HASHES[REFERENCE], "embedded reference hash differs")
    require(reference["ceiling"] == REFERENCE_CEILING, "reference ceiling changed")
    require(reference["disposition"] == REFERENCE_DISPOSITION, "reference disposition changed")

    require(len(production["acceptance_checks"]) == 24, "production check count changed")
    require(all(production["acceptance_checks"].values()), "production acceptance check failed")
    require(production["accepted_energy_constrained_numerical_return"], "production did not accept bounded return")
    require(len(reference["checks"]) == 13, "reference check count changed")
    require(all(reference["checks"].values()), "reference check failed")
    require(
        all(production["declared_named_fixture_convergence"].values()),
        "production named-fixture convergence failed",
    )
    require(
        all(reference["declared_cutoff_convergence"].values()),
        "reference named-fixture convergence failed",
    )


def algebra_program_and_boundary_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    ideal = production["ideal_infinite_ccr_law"]
    require(
        ideal["chronological_pulses"]
        == ["R_B(-mu)", "Q_A(-lambda)", "R_B(+mu)", "Q_A(+lambda)"],
        "production pulse order changed",
    )
    require(ideal["exact_factorization"] == "exp(-i lambda mu A B) tensor I_bus", "ideal sign changed")
    require(not ideal["finite_cutoff_uniform_arbitrary_state_claim"], "finite arbitrary-state claim enabled")

    programs = production["programs"]
    require(len(programs) == 2, "program count changed")
    expected = ((0, 1, 0.5, math.pi / 4.0, math.pi / 8.0), (1, 2, 2.0 / 3.0, math.pi / 4.0, math.pi / 6.0))
    for program, values in zip(programs, expected, strict=True):
        for key, value in zip(("a_client", "b_client", "lambda", "mu", "theta"), values, strict=True):
            close(program[key], value, absolute=2e-15, label=f"program {key}")

    model = reference["model"]
    require(model["cutoffs"] == list(CUTOFFS), "reference cutoffs changed")
    for label, program, query in (("A", programs[0], model["query_a"]), ("B", programs[1], model["query_b"])):
        for key in ("lambda", "mu", "theta"):
            close(program[key], query[key], absolute=2e-15, label=f"{label} reference {key}")

    compiled = production["direct_compiled_oracle"]["combined_boundary"]
    reference_compiled = reference["phase_sign_and_boundaries"]["direct_compiled_boundary_moments"]
    reference_evolved = reference["phase_sign_and_boundaries"]["combined_boundary_moments"]
    for name, exact in FINAL_BOUNDARY.items():
        close(compiled[name], exact, absolute=2e-12, label=f"production analytic {name}")
        close(reference_compiled[name], exact, absolute=2e-12, label=f"reference compiled {name}")
        close(reference_evolved[name], exact, absolute=2e-10, label=f"reference evolved {name}")

    require(production["direct_compiled_oracle"]["direct_zz_phase_gates"] == 2, "direct gate count changed")
    require(production["direct_compiled_oracle"]["weyl_conditional_pulses"] == 8, "Weyl pulse count changed")
    require(production["direct_compiled_oracle"]["software_forward_shadow_omits_bus_and_return"], "forward shadow changed")
    require(reference["phase_sign_and_boundaries"]["combined_client_trace_distance_to_correct_sign"] <= 2e-10, "reference correct sign failed")
    require(reference["phase_sign_and_boundaries"]["combined_client_trace_distance_to_opposite_sign"] >= 0.5, "reference opposite sign not rejected")


def finite_cutoff_and_custody_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    require(tuple(int(key) for key in production["cutoff_sweep"]) == (128, 16, 32, 64), "production serialized cutoff keys changed")
    require(set(reference["cutoff_sweep"]) == {str(value) for value in CUTOFFS}, "reference cutoffs missing")

    thresholds = production["thresholds"]
    for cutoff in CUTOFFS:
        fixtures = fixture_map(production, cutoff)
        require(set(fixtures) == PRODUCTION_FIXTURES, f"production fixtures changed at {cutoff}")
        require(set(reference["cutoff_sweep"][str(cutoff)]["fixtures"]) == REFERENCE_FIXTURES, f"reference fixtures changed at {cutoff}")

        commutator = production["finite_cutoff_commutator"][str(cutoff)]
        close(commutator["operator_norm_distance_from_infinite_ccr"], cutoff, absolute=1e-10, label=f"CCR defect {cutoff}")
        ref_top = reference["cutoff_sweep"][str(cutoff)]["top_fock_nonuniformity_control"]
        close(ref_top["ccr_defect_operator_norm"], cutoff, absolute=5e-12, label=f"reference CCR defect {cutoff}")

        top = production["top_fock_nonuniform_counterexample"][str(cutoff)]
        require(top["client_trace_distance_to_ideal"] >= 0.5, f"top client counterexample collapsed at {cutoff}")
        require(top["bus_trace_distance_to_supplied"] >= 0.5, f"top bus counterexample collapsed at {cutoff}")
        require(ref_top["top_fock_return_trace_distance"] >= 0.1, f"reference top witness collapsed at {cutoff}")

        for record in fixtures.values():
            custody = record["logical_resident_custody"]
            require(custody["allocation_object_unchanged"], "allocation object changed")
            require(custody["allocation_base_pointer_unchanged"], "allocation pointer changed")
            require(custody["carrier_supply_count"] == 1, "carrier supplied more than once")
            require(custody["program_count"] == 2, "program count changed")
            require(custody["generation_sequence"] == [0, 1, 2], "generation law changed")
            require(custody["privileged_midpoint_boundary_reads"] == 1, "midpoint read count changed")
            require(custody["client_detach_count"] == custody["client_replacement_count"] == 0, "client detached/replaced")
            require(custody["post_supply_carrier_state_set_count"] == 0, "carrier state reset")
            require(custody["snapshot_count"] == custody["reload_count"] == custody["reinitialize_count"] == 0, "accepted path used reset/reload")
            require(custody["guest_boundary_release_count"] == 1, "boundary release count changed")
            require(not custody["physical_same_mode_custody_established"], "physical custody promoted")
            require(len(record["process_diagnostics"]["per_pulse"]) == 8, "per-pulse trace length changed")

    final = fixture_map(production, 128)
    initial = fixture_map(production, 16)
    for name in (
        "coherent_alpha_0p65_plus_0p20i",
        "squeezed_r_0p45_phi_0p30",
        "thermal_nbar_0p4",
        "phi4_bus_reference",
    ):
        first_error = initial[name]["combined_a_then_b_released_boundary"][
            "complete_joint_to_compiled_client_tensor_supplied_br_frobenius"
        ]
        final_error = final[name]["combined_a_then_b_released_boundary"][
            "complete_joint_to_compiled_client_tensor_supplied_br_frobenius"
        ]
        require(final_error < first_error, f"production {name} did not converge")
    for record in final.values():
        first = record["program_a_privileged_nondestructive_boundary"]
        second = record["combined_a_then_b_released_boundary"]
        require(first["client_trace_distance_to_direct_compiler"] <= thresholds["final_client_trace_distance_max"], "A client parity failed")
        require(second["client_trace_distance_to_direct_compiler"] <= thresholds["final_client_trace_distance_max"], "B client parity failed")
        require(first["bus_trace_distance_to_supplied"] <= thresholds["final_bus_trace_distance_after_a_max"], "A bus return failed")
        require(second["bus_trace_distance_to_supplied"] <= thresholds["final_bus_trace_distance_after_b_max"], "B bus return failed")
        require(first["complete_joint_to_compiled_client_tensor_supplied_br_frobenius"] <= thresholds["complete_joint_factorization_frobenius_max"], "A full factorization failed")
        require(second["complete_joint_to_compiled_client_tensor_supplied_br_frobenius"] <= thresholds["complete_joint_factorization_frobenius_max"], "B full factorization failed")
        require(set(first["boundary"]) == {"X0", "X1", "X2", "Y0Z1", "Z0Y1", "X0X1"}, "A boundary leaked fields")
        require(set(second["boundary"]) == set(FINAL_BOUNDARY), "B boundary leaked fields")
        for key in ("client_density_integrity", "bus_density_integrity"):
            for density in (first[key], second[key]):
                require(density["trace_error"] <= thresholds["density_trace_error_max"], "density trace failed")
                require(density["hermiticity_max_abs"] <= thresholds["density_hermiticity_max"], "density Hermiticity failed")
                require(density["minimum_eigenvalue"] >= thresholds["density_min_eigenvalue_min"], "density PSD failed")

    require(
        sum(record["bus_reference_coherence"] is not None for record in final.values())
        == 1,
        "Phi4 coherence applicability changed",
    )
    require(final["phi4_bus_reference"]["reference_dimension"] == 4, "Phi4 reference dimension changed")
    phi = final["phi4_bus_reference"]["bus_reference_coherence"]
    require(phi is not None, "Phi4 coherence record absent")
    require(phi["after_a_trace_distance"] <= 1e-9 and phi["after_b_trace_distance"] <= 1e-9, "Phi4 BR return failed")
    require(phi["after_a_entanglement_infidelity"] <= 1e-9 and phi["after_b_entanglement_infidelity"] <= 1e-9, "Phi4 entanglement failed")


def controls_resources_and_nonclaims_audit(
    production: Mapping[str, Any], reference: Mapping[str, Any]
) -> None:
    controls = production["controls"]
    require(controls["reverse_rectangle"]["bus_trace_distance"] <= 1e-9, "reverse bus failed")
    require(controls["reverse_rectangle"]["sign_sensitive_separation_from_accepted"] >= 1.0, "reverse sign failed")
    require(controls["commuting_quadrature_sham"]["client_trace_distance_to_initial_plus"] <= 1e-9, "commuting sham not identity")
    require(abs(controls["commuting_quadrature_sham"]["Y0Z1"]) <= 1e-12, "commuting sham phase leaked")
    require(controls["omitted_final_pulse"]["bus_trace_distance"] >= 0.05, "omitted pulse not detected")

    snapshot = controls["snapshot_reload_sham"]
    require(snapshot["classification"] == "SNAPSHOT_RELOAD", "snapshot classification changed")
    require(snapshot["snapshot_count"] == snapshot["reload_count"] == 1, "snapshot counts changed")
    require(snapshot["native_restoration_count"] == 0, "snapshot promoted to restoration")

    dephase = controls["marginal_only_dephasing_sham"]
    close(dephase["bus_marginal_trace_distance"], 0.0, absolute=1e-12, label="dephased bus marginal")
    close(dephase["bus_reference_trace_distance"], 0.75, absolute=1e-12, label="dephased BR distance")
    close(dephase["entanglement_fidelity"], 0.25, absolute=1e-12, label="dephased fidelity")
    ref_dephase = reference["bus_reference_shams"]["phi4_dephasing"]
    close(ref_dephase["bus_marginal_trace_distance"], 0.0, absolute=1e-12, label="reference marginal")
    close(ref_dephase["bus_reference_trace_distance"], 0.75, absolute=1e-12, label="reference BR distance")

    require(controls["final_q_area_error_0p05"]["client_trace_distance_to_compiled"] >= 1e-4, "area client error absent")
    require(controls["final_q_area_error_0p05"]["bus_trace_distance"] >= 1e-4, "area bus error absent")
    require(controls["free_rotation_0p05_between_pulses"]["client_trace_distance_to_compiled"] >= 1e-3, "rotation client error absent")
    require(controls["free_rotation_0p05_between_pulses"]["bus_trace_distance"] >= 1e-3, "rotation bus error absent")
    require(controls["kerr_n_n_minus_1_0p02_between_pulses"]["client_trace_distance_to_compiled"] >= 1e-4, "Kerr client error absent")
    require(controls["kerr_n_n_minus_1_0p02_between_pulses"]["bus_trace_distance"] >= 1e-3, "Kerr bus error absent")
    require(production["midloop_causal_witness"]["pure_joint_mutual_information_numeric_nats"] >= 1.0, "midloop causal witness absent")

    ledger = production["resource_ledger"]
    require(ledger["physical_modes_in_model"] == 1 and ledger["client_qubits"] == 3, "model size changed")
    require(ledger["cutoffs"] == list(CUTOFFS), "resource cutoffs changed")
    require(ledger["conditional_pulses_total"] == 8, "pulse ledger changed")
    require(ledger["retained_dynamic_trajectory_history_complex_cells"] == 0, "dynamic history appeared")
    require(ledger["retained_inverse_history_entries"] == 0, "inverse history appeared")
    require(ledger["privileged_full_initial_density_baselines_retained_for_validation"], "baseline disclosure lost")
    require(not ledger["privileged_validation_baselines_readable_by_dynamics"], "dynamics gained baseline access")
    require(not ledger["history_free_complete_experiment_claim"], "history-free overclaim")
    require(ledger["linear_algebra_library_internal_scratch_not_instrumented"], "scratch caveat lost")
    require(ledger["joules_uninstantiated_without_hardware_mapping"], "joules caveat lost")
    require(ledger["bandwidth_and_pulse_duration_uninstantiated_without_hardware_mapping"], "bandwidth caveat lost")
    require(ledger["cutoff_resources"]["128"]["maximum_canonical_joint_backing_complex_cells"] == 131072, "maximum backing cells changed")

    comparison = production["classical_comparison"]
    require(comparison["best_fixture_comparator"] == "DIRECT_COMPILED_COMMUTING_ZZ_PHASE_AND_SELECTED_MOMENT_PRODUCT_FORMULAS", "comparator changed")
    require(not comparison["phase_qemu_emulator_has_resource_advantage"], "advantage promoted")
    require(not comparison["m257_escape_established"], "M257 escape promoted")
    require(comparison["growing_edge_family_bus_pulses"] == "4E", "bus scaling changed")
    require(comparison["growing_edge_family_direct_compiled_gates"] == "E", "direct scaling changed")

    for key in (
        "physical_execution",
        "physical_same_mode_restoration",
        "phase_native_client_architecture",
        "replace_the_bit_with_pi_established",
        "computational_advantage",
        "unbounded_compute",
    ):
        require(not production[key], f"production promoted {key}")
    require(production["m257_intact"], "production weakens M257")

    claims = reference["claims"]
    for key in (
        "finite_cutoff_exact_arbitrary_state_return",
        "uniform_arbitrary_state_return",
        "same_backing_restoration",
        "physical_restoration",
        "physical_execution",
        "computational_advantage",
        "m257_escape",
        "unbounded_compute",
        "bit_replaced_with_pi",
    ):
        require(not claims[key], f"reference promoted {key}")
    require(claims["restoration_classification"] == "NO_RESTORATION_CLAIM", "reference claims production custody")
    require(reference["resource_accounting"]["resource_advantage_claim"] is False, "reference claims advantage")


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
    algebra_program_and_boundary_audit(production, reference)
    finite_cutoff_and_custody_audit(production, reference)
    controls_resources_and_nonclaims_audit(production, reference)
    seal_audit(production_bytes, reference_bytes, arguments.write_seals)

    print(
        "PASS_STRICT_SCOPE M265_QUANTUM_WEYL_LOOP_BUS "
        "SCIENCE=SEPARATE_REFERENCE_PARITY "
        "RESTORATION=NUMERICAL_PHYSICAL_STATE_RESTORATION "
        "SCOPE=ENERGY_CONSTRAINED_LOGICAL_BACKING_ONLY "
        "RESOURCE=PACKAGE_SELF_REVIEW M257=INTACT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
