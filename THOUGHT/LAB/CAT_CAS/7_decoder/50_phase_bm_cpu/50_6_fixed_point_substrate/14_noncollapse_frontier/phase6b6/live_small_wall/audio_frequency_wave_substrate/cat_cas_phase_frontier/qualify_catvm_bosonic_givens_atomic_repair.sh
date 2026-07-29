#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_catvm_bosonic_givens_atomic_repair.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
source_file="$frontier_dir/catvm_bosonic_givens_atomic_repair.cpp"
service="$evidence_dir/catvm_bosonic_givens_atomic_repair"
result="$evidence_dir/result.json"
controller_stderr="$evidence_dir/controller.stderr"

mkdir -p "$evidence_dir"

g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$source_file" \
  -o "$service"

"$repo_root/.venv/bin/python" \
  "$frontier_dir/catvm_bosonic_givens_atomic_repair_controller.py" \
  "$service" \
  "$evidence_dir" \
  >"$result" 2>"$controller_stderr"

jq -e '
  .result == "PASS"
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .mode_custody.cross_mode_commands_denied
  and .mode_custody.monotone_nonce_replay_denied
  and .mode_custody.one_connection_lease
  and .hidden_intermediate.complex_cells == 4845
  and .hidden_intermediate.resident
  and .hidden_intermediate.content_derived_wire_hashes == 0
  and .hidden_intermediate.boundary_values_in_stage_response == 0
  and .hidden_intermediate.projection == "DENIED"
  and .atomic_ordering.final_boundary_after_restoration
  and .atomic_ordering.primary_generation == 1
  and .atomic_ordering.disconnect.stage_resident_before_disconnect
  and .atomic_ordering.disconnect.service_exit_after_inverse_cleanup == 0
  and .atomic_ordering.disconnect.internal_fresh_reuse_sentinel == "PASS"
  and (.atomic_ordering.disconnect.boundary_released | not)
  and .matched_arms.direct_restoration_class == "NO_RESTORATION_CLAIM"
  and .matched_arms.snapshot_restoration_class == "SNAPSHOT_RELOAD"
  and .matched_arms.in_place_restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .matched_arms.direct_boundary_error < 3e-11
  and .matched_arms.snapshot_boundary_error < 3e-11
  and .matched_arms.snapshot_reload_bytes == 4560
  and .primary.restoration_error < 3e-11
  and .primary.generation == 1
  and .primary.carrier_backing_preserved
  and .primary.snapshot_reload_bytes == 0
  and .reuse.restoration_error < 3e-11
  and .reuse.generation == 2
  and .reuse.fresh_restored_boundary_error < 3e-11
  and .reuse.same_backing
  and .reuse.same_native_operation_signature
  and .reuse.snapshot_reload_bytes == 0
  and .controls.missing_inverse_error > 1e-5
  and .controls.wrong_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .controls.failed_inverse_poisoned_service
  and .controls.null_carrier.backing_cells == 0
  and .controls.null_carrier.null_command == "DENIED"
  and .controls.null_carrier.in_place_command == "DENIED"
  and (.no_smuggle.controller_imports_backend | not)
  and (.no_smuggle.stage_boundary_valid | not)
  and .no_smuggle.service_stdout_bytes == 0
  and .no_smuggle.service_stderr_bytes == 0
  and .no_smuggle.receipt_depends_only_on_nonce_and_type
  and .resource_ceiling.carrier_cells == 285
  and .resource_ceiling.hidden_occupation_cells == 4845
  and .resource_ceiling.retained_inverse_history_bytes == 0
  and (.resource_ceiling.allocator_native_library_os_memory_bounded | not)
  and .matched_classical_recurrence_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.terminal | not)
' "$result" >/dev/null

test ! -s "$controller_stderr"

sha256sum \
  "$frontier_dir/four_rotor_bosonic_givens_phase.cpp" \
  "$frontier_dir/catvm_bosonic_givens_service_tail.inc" \
  "$source_file" \
  "$frontier_dir/catvm_bosonic_givens_protocol.py" \
  "$frontier_dir/catvm_bosonic_givens_atomic_repair_controller.py" \
  "$frontier_dir/qualify_catvm_bosonic_givens_atomic_repair.sh" \
  "$service" \
  "$result" \
  >"$evidence_dir/SHA256SUMS"

echo "CATVM bosonic Givens atomic repair qualification: PASS"
