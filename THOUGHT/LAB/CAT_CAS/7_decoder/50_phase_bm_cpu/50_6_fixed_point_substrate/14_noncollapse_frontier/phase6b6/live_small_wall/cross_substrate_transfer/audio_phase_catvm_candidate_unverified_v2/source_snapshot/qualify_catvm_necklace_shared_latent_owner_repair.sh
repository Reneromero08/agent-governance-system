#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo \
    "usage: qualify_catvm_necklace_shared_latent_owner_repair.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
direct_source="$frontier_dir/four_rotor_necklace_shared_latent_owner_repair.cpp"
service_source="$frontier_dir/catvm_necklace_shared_latent_owner_repair_service.cpp"
direct="$evidence_dir/four_rotor_necklace_shared_latent_owner_repair"
service="$evidence_dir/catvm_necklace_shared_latent_owner_repair_service"
direct_result="$evidence_dir/direct_owner_repair.json"
service_result="$evidence_dir/catvm_owner_repair.json"
direct_stderr="$evidence_dir/direct_owner_repair.stderr"
service_stderr="$evidence_dir/catvm_owner_repair.stderr"

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
  "$frontier_dir/catvm_necklace_shared_latent_owner_repair_controller.py" \
  "$service" \
  "$evidence_dir" \
  >"$service_result" 2>"$service_stderr"

test ! -s "$direct_stderr"
test ! -s "$service_stderr"

jq -e '
  .result == "PASS"
  and .declared_port_owner == 1279349809
  and .primary_module_owners_checked == 4
  and .reuse_module_owners_checked == 3
  and .wrong_nonzero_module_owner_rejected
  and .rejected_attack_carrier_error == 0
  and .primary_restoration_error < 6e-11
  and .reuse_restoration_error < 6e-11
  and .fresh_restored_reuse_boundary_error < 6e-11
  and .carrier_backing_preserved
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .predecessor_source_defect_preserved
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.terminal | not)
' "$direct_result" >/dev/null

jq -e '
  .result == "PASS"
  and .claim_ceiling
    == "LINUX_X86_64_SAME_UID_ONE_UNIX_SEQPACKET_CONNECTION_NONCE_DERIVED_OUTER_LEASE_EXACT_GENERATION_GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACES_570_COMPLEX_CELLS_TWO_CELL_LATENT_FIBER_FIXED_FOUR_MODULE_PRIMARY_THREE_MODULE_REUSE_STATIC_OWNER_0X4C415431_SEVEN_BIN_BOUNDARY_COMPLEX128_SOFTWARE_ONLY"
  and .module_port_owner.expected == 1279349809
  and .module_port_owner.wrong_nonzero_owner
    == "DENIED_BEFORE_CARRIER_OPERATION"
  and .module_port_owner.boundary_values_released == 0
  and .transaction_custody.exact_outer_lease
  and .transaction_custody.exact_outer_generation
  and .transaction_custody.post_attack_stage_resident
  and .transaction_custody.post_attack_final_generation == 1
  and .transaction_custody.post_attack_reuse_generation == 2
  and .primary_restoration_error < 6e-11
  and .reuse_restoration_error < 6e-11
  and .fresh_restored_reuse_boundary_error < 6e-11
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .predecessor_source_defect_preserved
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.terminal | not)
' "$service_result" >/dev/null

sha256sum \
  "$frontier_dir/four_rotor_necklace_shared_latent_phase.cpp" \
  "$frontier_dir/catvm_necklace_shared_latent_service.cpp" \
  "$direct_source" \
  "$service_source" \
  "$frontier_dir/catvm_necklace_shared_latent_owner_repair_controller.py" \
  "$frontier_dir/qualify_catvm_necklace_shared_latent_owner_repair.sh" \
  "$direct" \
  "$service" \
  "$direct_result" \
  "$service_result" \
  >"$evidence_dir/SHA256SUMS"

echo "CATVM shared-latent module-owner repair qualification: PASS"
