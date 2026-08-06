#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_two_shared_latent_phase.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
direct_source="$frontier_dir/four_rotor_necklace_two_shared_latent_phase.cpp"
service_source="$frontier_dir/catvm_necklace_two_shared_latent_service.cpp"
controller="$frontier_dir/catvm_necklace_two_shared_latent_controller.py"
oracle="$frontier_dir/two_shared_latent_phase_oracle.py"
durable_result="$frontier_dir/TWO_SHARED_LATENT_PHASE_RESULTS.json"
direct="$evidence_dir/four_rotor_necklace_two_shared_latent_phase"
service="$evidence_dir/catvm_necklace_two_shared_latent_service"
direct_result="$evidence_dir/direct.json"
service_result="$evidence_dir/catvm.json"
oracle_result="$evidence_dir/oracle.json"
direct_stderr="$evidence_dir/direct.stderr"
controller_stderr="$evidence_dir/controller.stderr"

mkdir -p "$evidence_dir"

g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$direct_source" \
  -o "$direct"
g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$service_source" \
  -o "$service"

nice -n 10 taskset -c 0-3 "$direct" \
  >"$direct_result" 2>"$direct_stderr"
"$repo_root/.venv/bin/python" \
  "$controller" \
  "$service" \
  "$evidence_dir/service-runs" \
  >"$service_result" 2>"$controller_stderr"
"$repo_root/.venv/bin/python" \
  "$oracle" \
  "$direct_result" \
  >"$oracle_result"

test ! -s "$direct_stderr"
test ! -s "$controller_stderr"

jq -e '
  .result == "PASS"
  and .resident_necklace_cells == 285
  and .latent_cells_per_necklace == 4
  and .resident_joint_complex_cells == 1140
  and .shared_latent_port_count == 2
  and .joint_consumer_count == 2
  and (.latent_ports_projected | not)
  and .relation_table_cells == 0
  and .assignment_cells == 0
  and .nonseparability_probe.product_input_maximum_fiber_determinant
    < 1e-15
  and .nonseparability_probe.post_joint_maximum_fiber_determinant
    > 1e-8
  and .nonseparability_probe.joint_inverse_restoration_error < 1.2e-10
  and .primary_restoration_error < 1.2e-10
  and .reuse_restoration_error < 1.2e-10
  and .fresh_restored_reuse_boundary_error < 1.2e-10
  and .carrier_backing_preserved
  and .controls.joint_module_deletion_boundary_effect > 1e-7
  and .controls.identity_joint_same_generator_boundary_effect > 1e-7
  and .controls.separable_joint_same_generator_boundary_effect > 1e-7
  and (
    [.controls.joint_consumer_strength_boundary_effects[]]
    | all(. > 1e-7)
  )
  and .controls.module_order_boundary_effect > 1e-7
  and .controls.semantic_perturbation_boundary_effect > 1e-7
  and .controls.missing_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .controls.wrong_semantic_inverse_error > 1e-5
  and .controls.wrong_type_rejected
  and .controls.wrong_owner_rejected
  and .controls.undermerge_boundary_effect > 1e-7
  and .resource_law.carrier_payload_bytes == 18240
  and .resource_law.persistent_baseline_plus_carrier_payload_bytes
    == 36480
  and .resource_law.declared_generator_and_extraction_scratch_payload_bytes
    == 18240
  and .resource_law.retained_inverse_history_bytes == 0
  and .strongest_compact_classical.same_1140_complex_recurrence_exists
  and .strongest_compact_classical.identity_reexecution_boundary_error
    == 0
  and .strongest_compact_classical.verification_level
    == "PACKAGE_SELF_REVIEW"
  and (.strongest_compact_classical.separate_reference_parity | not)
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and (.machine_boundary_enforced | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.general_catalytic_inference_established | not)
  and (.terminal | not)
' "$direct_result" >/dev/null

jq -e '
  .result == "PASS"
  and .custody.resident_joint_complex_cells == 1140
  and .custody.latent_cells_per_necklace == 4
  and .custody.shared_latent_port_count == 2
  and .custody.port_a.id == 1279349809
  and .custody.port_b.id == 1279349810
  and .custody.port_a.type != .custody.port_b.type
  and .custody.port_a.owner != .custody.port_b.owner
  and .custody.exact_outer_lease
  and .custody.exact_outer_generation
  and .custody.distinct_internal_leases
  and .custody.nonce_dependent_internal_leases
  and .custody.atomic_internal_generation
  and .custody.full_tuple_bound_per_module
  and .custody.reuse_full_tuple_bound_per_module
  and .custody.disconnect_reuse_sentinel_full_tuple_bound
  and .custody.stage_resident
  and (.custody.projected | not)
  and .atomic_ordering.final_boundary_after_restoration
  and .atomic_ordering.primary_generation == 1
  and .atomic_ordering.disconnect.inverse_cleanup_exit == 0
  and (.atomic_ordering.disconnect.response_boundary_released | not)
  and .atomic_ordering.staged_stop.acknowledgement_after_rollback
  and (.atomic_ordering.staged_stop.boundary_released | not)
  and (.atomic_ordering.staged_stop.generation_advanced | not)
  and .atomic_ordering.denied_staged_stop
    .wrong_lease_denied_without_termination
  and .atomic_ordering.denied_staged_stop
    .wrong_generation_denied_without_termination
  and .atomic_ordering.denied_staged_stop.subsequent_final_restored
  and .primary.restoration_error < 1.2e-10
  and .primary.baseline_reload_bytes == 0
  and .reuse.restoration_error < 1.2e-10
  and .reuse.fresh_restored_boundary_error < 1.2e-10
  and .reuse.generation == 2
  and .reuse.same_backing
  and .reuse.same_generator_term_count_and_carrier_shape
  and .reuse.baseline_reload_bytes == 0
  and .controls.missing_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .controls.wrong_semantic_inverse_error > 1e-5
  and .controls.wrong_type == "DENIED_BEFORE_CARRIER_OPERATION"
  and .controls.wrong_owner_a == "DENIED_BEFORE_CARRIER_OPERATION"
  and .controls.wrong_owner_b == "DENIED_BEFORE_CARRIER_OPERATION"
  and .controls.undermerge == "DENIED_BEFORE_CARRIER_OPERATION"
  and .controls.duplicate_port == "DENIED_BEFORE_CARRIER_OPERATION"
  and .controls.stale_internal_generation
    == "DENIED_BEFORE_CARRIER_OPERATION"
  and .controls.wrong_internal_lease
    == "DENIED_BEFORE_CARRIER_OPERATION"
  and .controls.premature_projection == "DENIED"
  and .controls.snapshot == "DENIED"
  and .controls.null_carrier.backing_cells == 0
  and .no_smuggle.controller_imports_backend == false
  and .no_smuggle.stage_boundary_values == 0
  and .no_smuggle.latent_values_in_response == 0
  and .no_smuggle.content_derived_receipts == 0
  and .no_smuggle.stdout_bytes == 0
  and .no_smuggle.stderr_bytes == 0
  and .resource_law.carrier_payload_bytes == 18240
  and .resource_law.persistent_baseline_plus_carrier_payload_bytes
    == 36480
  and .resource_law.reuse_verification_carrier_payload_bytes == 18240
  and .resource_law.request_bytes == 32
  and .resource_law.response_bytes == 116
  and .resource_law.retained_inverse_history_bytes == 0
  and .strongest_compact_classical.identical_1140_complex_recurrence_exists
  and .strongest_compact_classical.verification_level
    == "PACKAGE_SELF_REVIEW"
  and (.strongest_compact_classical.separate_reference_parity | not)
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.general_catalytic_inference_established | not)
  and (.terminal | not)
' "$service_result" >/dev/null

jq -e '
  .result == "PASS"
  and (.production_backend_imported | not)
  and (.production_recurrence_reimplemented | not)
  and .checks.production_passed
  and .checks.production_joint_cells
  and .checks.production_two_ports
  and .checks.production_probe_nonzero
  and .checks.product_determinant_zero
  and .checks.joint_determinant_nonzero
  and .checks.controlled_phase_nonfactorable
  and .checks.local_joint_noncommuting
  and .checks.joint_inverse_restores
  and (.terminal | not)
' "$oracle_result" >/dev/null

jq -e \
  --slurpfile direct "$direct_result" \
  --slurpfile catvm "$service_result" \
  --slurpfile oracle "$oracle_result" \
  '
    .result == "PASS"
    and .classification == "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL"
    and .verification_level == "CLEANROOM_ADVERSARIAL_VERIFICATION"
    and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
    and .carrier.resident_complex_cells
      == $direct[0].resident_joint_complex_cells
    and .direct.primary_boundary == $direct[0].boundary
    and .direct.primary_restoration_error
      == $direct[0].primary_restoration_error
    and .direct.reuse_restoration_error
      == $direct[0].reuse_restoration_error
    and .direct.fresh_restored_reuse_boundary_error
      == $direct[0].fresh_restored_reuse_boundary_error
    and .direct.nonseparability_probe
      == $direct[0].nonseparability_probe
    and .direct.controls == $direct[0].controls
    and .catvm.primary_restoration_error
      == $catvm[0].primary.restoration_error
    and .catvm.reuse_restoration_error
      == $catvm[0].reuse.restoration_error
    and .catvm.fresh_restored_reuse_boundary_error
      == $catvm[0].reuse.fresh_restored_boundary_error
    and .catvm.primary_generation
      == $catvm[0].atomic_ordering.primary_generation
    and .catvm.reuse_generation == $catvm[0].reuse.generation
    and .independent_algebra_oracle.product_input_determinant
      == $oracle[0].product_input_determinant
    and .independent_algebra_oracle.post_joint_determinant
      == $oracle[0].post_joint_determinant
    and .independent_algebra_oracle
      .controlled_phase_factorization_residual
      == $oracle[0].controlled_phase_factorization_residual
    and .independent_algebra_oracle
      .local_joint_noncommutation_distance
      == $oracle[0].local_joint_noncommutation_distance
    and (.claim_limits.distinct_phase_resource_established | not)
    and (.claim_limits.computational_advantage | not)
    and (.claim_limits.small_wall_crossed | not)
    and (.claim_limits.physical_waveform_execution | not)
    and (.terminal | not)
  ' "$durable_result" >/dev/null

if rg -n \
  'ctypes|dlopen|four_rotor_necklace_two_shared_latent_phase' \
  "$controller"; then
  echo "controller contains a prohibited backend-loading path" >&2
  exit 1
fi

jq -n \
  --slurpfile direct "$direct_result" \
  --slurpfile catvm "$service_result" \
  --slurpfile oracle "$oracle_result" \
  '{
    result: "PASS",
    direct: $direct[0],
    catvm: $catvm[0],
    independent_algebra_oracle: $oracle[0],
    classification: "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL",
    verification_level: "CLEANROOM_ADVERSARIAL_VERIFICATION",
    restoration_class: "NUMERICAL_PHYSICAL_STATE_RESTORATION",
    claim_ceiling: (
      "BOUNDED_GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_"
      + "ROTORS_285_NECKLACES_FOUR_CELL_TWO_BINARY_LATENT_FIBER_"
      + "TWO_EXACT_FULL_TUPLE_BOUND_PORTS_FIXED_SIX_MODULE_PRIMARY_"
      + "FOUR_MODULE_REUSE_TWO_CONTROLLED_PHASE_JOINT_CONSUMERS_"
      + "IDENTITY_AND_ONE_DECLARED_SEPARABLE_SAME_GENERATOR_CONTROL_"
      + "LINUX_X86_64_SAME_UID_ONE_UNIX_SEQPACKET_CONNECTION_"
      + "FINAL_SEVEN_BIN_NORM_BOUNDARY_COMPLEX128_SOFTWARE_ONLY"
    ),
    exhaustive_separable_lower_bound_established: false,
    generic_two_port_algebra_established: false,
    separate_classical_reference_parity: false,
    distinct_phase_resource_established: false,
    computational_advantage: false,
    small_wall_crossed: false,
    physical_waveform_execution: false,
    general_catalytic_inference_established: false,
    terminal: false
  }' >"$evidence_dir/combined.json"

sha256sum \
  "$direct_source" \
  "$service_source" \
  "$controller" \
  "$oracle" \
  "$frontier_dir/qualify_two_shared_latent_phase.sh" \
  "$direct" \
  "$service" \
  "$direct_result" \
  "$service_result" \
  "$oracle_result" \
  "$evidence_dir/combined.json" \
  >"$evidence_dir/SHA256SUMS"

echo "two-shared-latent phase qualification: PASS"
