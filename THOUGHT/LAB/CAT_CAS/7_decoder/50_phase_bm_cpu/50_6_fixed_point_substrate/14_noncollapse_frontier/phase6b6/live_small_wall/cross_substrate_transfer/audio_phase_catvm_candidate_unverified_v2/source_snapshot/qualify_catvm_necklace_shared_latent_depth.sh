#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo \
    "usage: qualify_catvm_necklace_shared_latent_depth.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
service_source="$frontier_dir/catvm_necklace_shared_latent_depth_service.cpp"
service="$evidence_dir/catvm_necklace_shared_latent_depth_service"
catvm_result="$evidence_dir/catvm_result.json"
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
  "$frontier_dir/catvm_necklace_shared_latent_depth_controller.py" \
  "$service" \
  "$evidence_dir" \
  >"$catvm_result" 2>"$controller_stderr"
test ! -s "$controller_stderr"

jq -e '
  .result == "PASS"
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .custody.resident_joint_complex_cells == 570
  and .custody.program_descriptor_bytes == 8
  and .custody.retained_module_tape_bytes == 0
  and .custody.retained_inverse_history_bytes == 0
  and .custody.primary_depth == 32
  and .custody.stage_applied_modules == 16
  and .custody.stage_resident
  and (.custody.projected | not)
  and .custody.wrong_owner_denied
  and .custody.wrong_lease_denied
  and .custody.wrong_generation_denied
  and .custody.typed_receipt_only
  and .atomic_ordering.final_boundary_after_restoration
  and .atomic_ordering.primary_generation == 1
  and .atomic_ordering.disconnect.stage_depth == 8
  and .atomic_ordering.disconnect.stage_resident
  and (.atomic_ordering.disconnect.boundary_released | not)
  and .atomic_ordering.disconnect.inverse_cleanup_exit == 0
  and .atomic_ordering.disconnect.internal_fresh_reuse_sentinel == "PASS"
  and .primary.restoration_error < 6e-11
  and .primary.carrier_backing_preserved
  and .primary.baseline_reload_bytes == 0
  and .reuse.depth == 11
  and .reuse.restoration_error < 6e-11
  and .reuse.fresh_restored_boundary_error < 6e-11
  and .reuse.generation == 2
  and .reuse.same_backing
  and .reuse.same_resource_signature
  and .reuse.baseline_reload_bytes == 0
  and .controls.missing_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .controls.wrong_inverse_variant_error > 1e-5
  and .controls.wrong_owner == "DENIED"
  and .controls.premature_projection == "DENIED"
  and .controls.snapshot == "DENIED"
  and .controls.null_carrier.backing_cells == 0
  and .controls.null_carrier.null_command == "DENIED"
  and .controls.null_carrier.begin_command == "ERROR_BEFORE_ACCESS"
  and .controls.poison_after_failed_inverse
  and (.no_smuggle.controller_imports_backend | not)
  and .no_smuggle.stage_boundary_values == 0
  and .no_smuggle.latent_values_in_response == 0
  and .no_smuggle.content_derived_receipts == 0
  and .no_smuggle.stdout_bytes == 0
  and .no_smuggle.stderr_bytes == 0
  and .resource_law.resident_joint_complex_cells == 570
  and .resource_law.public_pending_topology_bytes == 8
  and .resource_law.pending_applied_counter_bytes == 8
  and .resource_law.relation_table_cells == 0
  and .resource_law.assignment_cells == 0
  and .resource_law.temporary_occupation_cells == 0
  and .resource_law.dense_285_operator_cells == 0
  and .resource_law.retained_inverse_history_bytes == 0
  and .resource_law.inverse_descriptors_rematerialized
  and .strongest_compact_classical_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$catvm_result" >/dev/null

jq -e -n \
  --slurpfile direct "$evidence_dir/direct/result.json" \
  --slurpfile catvm "$catvm_result" \
  '[
    range(0; 7)
    | (($direct[0].strongest_compact_classical.boundary_error == 0)
       and ($catvm[0].primary.boundary[.] >= 0))
  ] | all' >/dev/null

if rg -q 'four_rotor_necklace_shared_latent_depth_compiler|depth_service' \
    "$frontier_dir/catvm_necklace_shared_latent_depth_controller.py"; then
  echo "controller imported or named backend implementation" >&2
  exit 1
fi

sha256sum \
  "$frontier_dir/four_rotor_necklace_shared_latent_depth_compiler.cpp" \
  "$service_source" \
  "$frontier_dir/catvm_necklace_shared_latent_depth_protocol.py" \
  "$frontier_dir/catvm_necklace_shared_latent_depth_controller.py" \
  "$frontier_dir/qualify_four_rotor_necklace_shared_latent_depth_compiler.sh" \
  "$frontier_dir/qualify_catvm_necklace_shared_latent_depth.sh" \
  "$service" \
  "$evidence_dir/direct/result.json" \
  "$catvm_result" \
  >"$evidence_dir/SHA256SUMS"

echo "CATVM shared-latent depth qualification: PASS"
