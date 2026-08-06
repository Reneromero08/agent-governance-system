#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_catvm_necklace_shared_latent.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
direct_source="$frontier_dir/four_rotor_necklace_shared_latent_phase.cpp"
service_source="$frontier_dir/catvm_necklace_shared_latent_service.cpp"
direct="$evidence_dir/four_rotor_necklace_shared_latent_phase"
service="$evidence_dir/catvm_necklace_shared_latent_service"
direct_result="$evidence_dir/direct_result.json"
catvm_result="$evidence_dir/catvm_result.json"
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
  "$frontier_dir/catvm_necklace_shared_latent_controller.py" \
  "$service" \
  "$evidence_dir" \
  >"$catvm_result" 2>"$controller_stderr"

test ! -s "$direct_stderr"
test ! -s "$controller_stderr"

jq -e '
  .result == "PASS"
  and .resident_necklace_cells == 285
  and .latent_cells_per_necklace == 2
  and .resident_joint_complex_cells == 570
  and .shared_latent_port_count == 1
  and .shared_latent_consumers == 4
  and .noncommuting_axes == ["Z","X","Y","X"]
  and (.public_scalar_observation_substitution | not)
  and (.latent_port_projected | not)
  and .relation_table_cells == 0
  and .assignment_cells == 0
  and .primary_restoration_error < 6e-11
  and .reuse_restoration_error < 6e-11
  and .fresh_restored_reuse_boundary_error < 6e-11
  and .carrier_backing_preserved
  and .restoration_generation == 2
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .controls.overmerge_boundary_effect > 1e-5
  and .controls.undermerge_boundary_effect > 1e-5
  and .controls.module_order_boundary_effect > 1e-5
  and .controls.semantic_perturbation_boundary_effect > 1e-5
  and .controls.missing_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .controls.wrong_semantic_inverse_error > 1e-5
  and .controls.wrong_type_rejected
  and .resource_law.temporary_occupation_cells == 0
  and .resource_law.dense_285_operator_cells == 0
  and .resource_law.retained_inverse_history_bytes == 0
  and .strongest_compact_classical.same_570_complex_recurrence
  and .strongest_compact_classical.boundary_error == 0
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
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .custody.resident_joint_complex_cells == 570
  and .custody.shared_latent_cells_per_necklace == 2
  and .custody.shared_latent_consumers == 4
  and .custody.stage_resident
  and (.custody.projected | not)
  and .custody.wrong_owner_denied
  and .custody.wrong_generation_denied
  and .custody.typed_receipt_only
  and .atomic_ordering.final_boundary_after_restoration
  and .atomic_ordering.primary_generation == 1
  and .atomic_ordering.disconnect.stage_resident
  and (.atomic_ordering.disconnect.response_boundary_released | not)
  and .atomic_ordering.disconnect.inverse_cleanup_exit == 0
  and .atomic_ordering.disconnect.internal_fresh_reuse_sentinel == "PASS"
  and .primary.restoration_error < 6e-11
  and .primary.carrier_backing_preserved
  and .primary.baseline_reload_bytes == 0
  and .reuse.restoration_error < 6e-11
  and .reuse.fresh_restored_boundary_error < 6e-11
  and .reuse.generation == 2
  and .reuse.same_backing
  and .reuse.same_resource_signature
  and .reuse.baseline_reload_bytes == 0
  and .controls.missing_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .controls.wrong_semantic_inverse_error > 1e-5
  and .controls.wrong_type == "DENIED"
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
  and .resource_law.relation_table_cells == 0
  and .resource_law.assignment_cells == 0
  and .resource_law.temporary_occupation_cells == 0
  and .resource_law.dense_285_operator_cells == 0
  and .resource_law.retained_inverse_history_bytes == 0
  and .strongest_compact_classical_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.general_catalytic_inference_established | not)
  and (.terminal | not)
' "$catvm_result" >/dev/null

jq -e -n \
  --slurpfile direct "$direct_result" \
  --slurpfile catvm "$catvm_result" \
  '[
    range(0; 7)
    | (($direct[0].boundary[.] - $catvm[0].primary.boundary[.]) | abs)
  ] | max < 6e-11' >/dev/null

sha256sum \
  "$frontier_dir/four_rotor_necklace_generator_phase.cpp" \
  "$direct_source" \
  "$service_source" \
  "$frontier_dir/catvm_necklace_shared_latent_protocol.py" \
  "$frontier_dir/catvm_necklace_shared_latent_controller.py" \
  "$frontier_dir/qualify_catvm_necklace_shared_latent.sh" \
  "$direct" \
  "$service" \
  "$direct_result" \
  "$catvm_result" \
  >"$evidence_dir/SHA256SUMS"

echo "CATVM shared-latent necklace qualification: PASS"
