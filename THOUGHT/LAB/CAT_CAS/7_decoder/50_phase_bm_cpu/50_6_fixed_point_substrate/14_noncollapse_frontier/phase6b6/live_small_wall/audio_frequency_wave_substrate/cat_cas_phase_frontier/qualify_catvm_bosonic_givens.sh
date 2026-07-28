#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_catvm_bosonic_givens.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
base_source="$frontier_dir/four_rotor_bosonic_givens_phase.cpp"
tail_source="$frontier_dir/catvm_bosonic_givens_service_tail.inc"
generated_source="$evidence_dir/catvm_bosonic_givens_service.cpp"
service="$evidence_dir/catvm_bosonic_givens_service"
socket_path="$evidence_dir/catvm.sock"
result="$evidence_dir/result.json"
controller_stderr="$evidence_dir/controller.stderr"
service_stdout="$evidence_dir/service.stdout"
service_stderr="$evidence_dir/service.stderr"

mkdir -p "$evidence_dir"
sed 's/^int BOSONIC_GIVENS_ENTRY() {/int bosonic_givens_standalone_main() {/' \
  "$base_source" >"$generated_source"
printf '\n' >>"$generated_source"
sed '/^#include /d' "$tail_source" >>"$generated_source"

g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  -include cerrno \
  -include csignal \
  -include cstring \
  -include string \
  -include sys/prctl.h \
  -include sys/socket.h \
  -include sys/stat.h \
  -include sys/types.h \
  -include sys/un.h \
  -include unistd.h \
  -I "$frontier_dir" \
  "$generated_source" \
  -o "$service"

nice -n 10 taskset -c 0-3 "$service" "$socket_path" \
  >"$service_stdout" 2>"$service_stderr" &
service_pid=$!
cleanup() {
  if kill -0 "$service_pid" 2>/dev/null; then
    kill "$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 200); do
  [[ -S "$socket_path" ]] && break
  sleep 0.01
done
[[ -S "$socket_path" ]]

nice -n 10 taskset -c 0-3 \
  "$repo_root/.venv/bin/python" \
  "$frontier_dir/catvm_bosonic_givens_controller.py" \
  "$socket_path" \
  >"$result" 2>"$controller_stderr"

wait "$service_pid"
trap - EXIT

jq -e '
  .result == "PASS"
  and .machine_boundary.transport == "AF_UNIX_SOCK_SEQPACKET"
  and .machine_boundary.same_uid_peer_credential_gate
  and (.machine_boundary.backend_dumpable | not)
  and (.machine_boundary.controller_imports_phase_backend | not)
  and .hidden_intermediate.complex_cells == 4845
  and .hidden_intermediate.retained_across_protocol_boundary
  and .hidden_intermediate.continuation_consumed_actual_resident_intermediate
  and (.hidden_intermediate.decoded_or_serialized | not)
  and .hidden_intermediate.projection_status == "DENIED"
  and (.hidden_intermediate.projection_boundary_valid | not)
  and .matched_arms.direct_boundary_error < 3e-11
  and .matched_arms.snapshot_boundary_error < 3e-11
  and .matched_arms.direct_arm_scope == "SERVICE_LOCAL_FORWARD_ONLY_MATCHED_PHASE_BASELINE"
  and (.matched_arms.warm_direct_process_baseline_established | not)
  and .matched_arms.snapshot_creation_bytes == 4560
  and .matched_arms.snapshot_reload_bytes == 4560
  and (.matched_arms.snapshot_is_accepted_restoration | not)
  and .matched_arms.packets_per_matched_arm == 2
  and .matched_arms.logical_protocol_bytes_per_matched_arm == 280
  and .primary.restoration_error < 3e-11
  and .primary.restoration_generation == 1
  and .primary.actual_inverse_restoration
  and .primary.carrier_backing_preserved
  and .primary.boundary_retained_after_backend_restoration
  and .primary.snapshot_reload_bytes == 0
  and .primary.resources.carrier_payload_bytes == 4560
  and .primary.resources.hidden_occupation_bytes == 77520
  and .primary.resources.compiled_plan_conservative_payload_bytes == 19867
  and .primary.resources.maximum_service_explicit_payload_bytes == 102007
  and .primary.resources.maximum_service_plus_packet_payload_bytes == 102147
  and .primary.resources.retained_inverse_history_bytes == 0
  and (.primary.resources.kernel_socket_buffer_payload_bounded | not)
  and (.primary.resources.host_allocator_metadata_bounded | not)
  and .reuse.restoration_error < 3e-11
  and .reuse.restoration_generation == 2
  and .reuse.actual_restored_carrier_reuse
  and .controls.attempted_projection == "DENIED"
  and .controls.null_carrier == "DENIED"
  and .controls.missing_inverse_error > 1e-5
  and .controls.wrong_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .no_smuggle.intermediate_complex_values_in_protocol == 0
  and .no_smuggle.intermediate_bytes_in_protocol == 0
  and (.no_smuggle.controller_computed_boundary_independently | not)
  and (.no_smuggle.ordinary_output_schema_contains_intermediate | not)
  and .no_smuggle.backend_queue_empty_after_transaction
  and .matched_classical_bosonic_givens_identical
  and (.cross_uid_secrecy_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$result" >/dev/null

test ! -s "$controller_stderr"
test ! -s "$service_stdout"
test ! -s "$service_stderr"

sha256sum \
  "$base_source" \
  "$tail_source" \
  "$frontier_dir/catvm_bosonic_givens_protocol.py" \
  "$frontier_dir/catvm_bosonic_givens_controller.py" \
  "$frontier_dir/qualify_catvm_bosonic_givens.sh" \
  "$generated_source" \
  "$service" \
  "$result" \
  >"$evidence_dir/SHA256SUMS"

echo "CATVM bosonic Givens qualification: PASS"
