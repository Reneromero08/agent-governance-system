#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo \
    "usage: qualify_catvm_necklace_shared_latent_depth_accounting_repair.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
service_source="$frontier_dir/catvm_necklace_shared_latent_depth_atomic_repair_service.cpp"
controller="$frontier_dir/catvm_necklace_shared_latent_depth_accounting_repair_controller.py"
service="$evidence_dir/catvm_necklace_shared_latent_depth_accounting_repair_service"
result="$evidence_dir/result.json"
controller_stderr="$evidence_dir/controller.stderr"

mkdir -p "$evidence_dir"

"$frontier_dir/qualify_four_rotor_necklace_shared_latent_depth_compiler.sh" \
  "$evidence_dir/direct"

g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$service_source" \
  -o "$service"

"$repo_root/.venv/bin/python" \
  "$controller" \
  "$service" \
  "$evidence_dir" \
  >"$result" 2>"$controller_stderr"
test ! -s "$controller_stderr"

jq -e '
  .result == "PASS"
  and .claim_candidate
      == "CATVM_ATOMIC_DISPATCH_AND_RESOURCE_ACCOUNTING_REPAIRED_TOPOLOGY_REMATERIALIZED_OWNER_BOUND_SHARED_LATENT_PHASE_PROGRAM_FIXED_570_CARRIER_AT_DEPTH32"
  and .verification_level == "PACKAGE_SELF_REVIEW"
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .historical_predecessor.classification == "REJECTED_SOURCE_DEFECT"
  and .historical_atomic_repair_package.classification
      == "REJECTED_SOURCE_DEFECT"
  and .historical_atomic_repair_package.source_defect
      == "LIVE_289_COMPLEX_GENERATOR_MATRIX_OMITTED_FROM_PRIMARY_AND_REUSE_PEAKS"
  and .historical_atomic_repair_package.preserved_subclaim
      == "ATOMIC_DISPATCH_INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .custody.resident_joint_complex_cells == 570
  and .custody.program_descriptor_bytes == 8
  and .custody.retained_module_tape_bytes == 0
  and .custody.retained_inverse_history_bytes == 0
  and .custody.primary_depth == 32
  and .custody.stage_applied_modules == 16
  and .custody.stage_resident
  and (.custody.projected | not)
  and .atomic_ordering.final_boundary_after_restoration
  and .atomic_ordering.primary_generation == 1
  and .atomic_ordering.disconnect.stage_resident
  and (.atomic_ordering.disconnect.boundary_released | not)
  and .atomic_ordering.disconnect.inverse_cleanup_exit == 0
  and .atomic_ordering.initialize_replay.repeated_initialize
      == "DENIED_WITHOUT_NONCE_MUTATION"
  and .atomic_ordering.initialize_replay.accepted_nonce_replay
      == "DENIED_BY_OWNER_GATE"
  and .atomic_ordering.depth_mismatch.mismatch_preserved_stage_receipt
  and .atomic_ordering.depth_mismatch.mismatch_boundary_values == 0
  and .atomic_ordering.depth_mismatch.staged_stop
      == "DENIED_WITH_STAGE_RESIDENT"
  and .atomic_ordering.variant_mismatch.mismatch_preserved_stage_receipt
  and .atomic_ordering.variant_mismatch.mismatch_boundary_values == 0
  and .atomic_ordering.variant_mismatch.staged_stop
      == "DENIED_WITH_STAGE_RESIDENT"
  and .primary.restoration_error < 6e-11
  and .primary.begin_native_generator_terms == 33736704
  and .primary.continue_and_inverse_native_generator_terms == 101210112
  and .primary.total_native_generator_terms == 134946816
  and .reuse.restoration_error < 6e-11
  and .reuse.fresh_restored_boundary_error < 6e-11
  and .reuse.generation == 2
  and .reuse.same_backing
  and .reuse.baseline_reload_bytes == 0
  and .controls.missing_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .controls.wrong_inverse_variant_error > 1e-5
  and .controls.wrong_owner == "DENIED"
  and .controls.premature_projection == "DENIED"
  and .controls.snapshot == "DENIED"
  and .controls.null_carrier.backing_cells == 0
  and .controls.invalid_topology.zero_depth == "ERROR_BEFORE_STAGE"
  and .controls.invalid_topology.zero_variant == "ERROR_BEFORE_STAGE"
  and .controls.invalid_topology.depth_above_ceiling
      == "ERROR_BEFORE_STAGE"
  and .resource_law.catalytic_carrier_complex_cells == 570
  and .resource_law.permanent_restoration_baseline_complex_cells == 570
  and .resource_law.per_module_fiber_complex_cells == 285
  and .resource_law.generator_matrix_complex_cells == 289
  and .resource_law.generator_plan_object_bytes == 4664
  and .resource_law.generator_plan_nonmatrix_scalar_bytes == 40
  and .resource_law.generator_work_complex_cells == 855
  and .resource_law.reuse_reference_complex_cells == 570
  and .resource_law.primary_peak_counted_complex_cells_excluding_plan == 2569
  and .resource_law.reuse_peak_counted_complex_cells_excluding_plan == 3139
  and .resource_law.plan_root_complex_cells == 17
  and .resource_law.primary_peak_counted_complex_cells_including_plan_roots
      == 2586
  and .resource_law.reuse_peak_counted_complex_cells_including_plan_roots
      == 3156
  and .resource_law.plan_object_bytes == 296
  and .resource_law.plan_necklace_count == 285
  and .resource_law.plan_necklace_capacity == 285
  and .resource_law.plan_necklace_element_bytes == 36
  and .resource_law.plan_necklace_capacity_bytes == 10260
  and .resource_law.request_packet_bytes == 32
  and .resource_law.response_packet_bytes == 116
  and .resource_law.accepted_primary_reuse_requests == 5
  and .resource_law.accepted_primary_reuse_request_bytes == 160
  and .resource_law.accepted_primary_reuse_response_bytes == 580
  and .resource_law.accepted_primary_reuse_protocol_bytes == 740
  and .resource_law.retained_inverse_history_bytes == 0
  and .resource_law.inverse_descriptors_rematerialized
  and (.no_smuggle.controller_imports_backend | not)
  and .no_smuggle.stage_boundary_values == 0
  and .no_smuggle.latent_values_in_response == 0
  and .no_smuggle.stdout_bytes == 0
  and .no_smuggle.stderr_bytes == 0
  and .strongest_compact_classical_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$result" >/dev/null

jq -e -n \
  --slurpfile direct "$evidence_dir/direct/result.json" \
  --slurpfile repaired "$result" \
  '
    ($direct[0].resource_law.generator_matrix_complex_cells
      == $repaired[0].resource_law.generator_matrix_complex_cells)
    and
    ($direct[0].resource_law.accepted_transaction_peak_counted_complex_cells_excluding_plan
      == $repaired[0].resource_law.primary_peak_counted_complex_cells_excluding_plan)
    and
    ($direct[0].resource_law.verification_peak_counted_complex_cells_excluding_plan
      == $repaired[0].resource_law.reuse_peak_counted_complex_cells_excluding_plan)
    and
    ($direct[0].resource_law.plan_root_complex_cells
      == $repaired[0].resource_law.plan_root_complex_cells)
    and
    ($direct[0].resource_law.plan_object_bytes
      == $repaired[0].resource_law.plan_object_bytes)
    and
    ($direct[0].resource_law.plan_necklace_count
      == $repaired[0].resource_law.plan_necklace_count)
    and
    ($direct[0].resource_law.plan_necklace_capacity
      == $repaired[0].resource_law.plan_necklace_capacity)
    and
    ($direct[0].resource_law.plan_necklace_element_bytes
      == $repaired[0].resource_law.plan_necklace_element_bytes)
    and
    ($direct[0].resource_law.plan_necklace_capacity_bytes
      == $repaired[0].resource_law.plan_necklace_capacity_bytes)
    and $direct[0].strongest_compact_classical.same_570_complex_recurrence
    and ($direct[0].strongest_compact_classical.boundary_error == 0)
  ' >/dev/null

if rg -q \
    'service\\.cpp|four_rotor_necklace_shared_latent_depth_compiler' \
    "$controller"; then
  echo "accounting repair controller imported backend source" >&2
  exit 1
fi

sha256sum \
  "$frontier_dir/four_rotor_necklace_shared_latent_depth_compiler.cpp" \
  "$frontier_dir/catvm_necklace_shared_latent_depth_service.cpp" \
  "$service_source" \
  "$frontier_dir/catvm_necklace_shared_latent_depth_protocol.py" \
  "$frontier_dir/catvm_necklace_shared_latent_depth_controller.py" \
  "$frontier_dir/catvm_necklace_shared_latent_depth_atomic_repair_controller.py" \
  "$controller" \
  "$frontier_dir/qualify_four_rotor_necklace_shared_latent_depth_compiler.sh" \
  "$frontier_dir/qualify_catvm_necklace_shared_latent_depth_atomic_repair.sh" \
  "$frontier_dir/qualify_catvm_necklace_shared_latent_depth_accounting_repair.sh" \
  "$service" \
  "$evidence_dir/direct/result.json" \
  "$evidence_dir/historical_atomic_result.json" \
  "$result" \
  >"$evidence_dir/SHA256SUMS"

echo "CATVM shared-latent depth accounting repair qualification: PASS"
