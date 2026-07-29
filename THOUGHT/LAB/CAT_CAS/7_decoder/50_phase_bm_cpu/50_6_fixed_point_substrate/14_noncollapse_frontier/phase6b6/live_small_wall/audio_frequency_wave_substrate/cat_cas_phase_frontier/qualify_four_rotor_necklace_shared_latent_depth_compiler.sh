#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo \
    "usage: qualify_four_rotor_necklace_shared_latent_depth_compiler.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_path="$frontier_dir/four_rotor_necklace_shared_latent_depth_compiler.cpp"
binary="$evidence_dir/four_rotor_necklace_shared_latent_depth_compiler"
result="$evidence_dir/result.json"
stderr_path="$evidence_dir/stderr.txt"

mkdir -p "$evidence_dir"

g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$source_path" \
  -o "$binary"

nice -n 10 taskset -c 0-3 "$binary" \
  >"$result" 2>"$stderr_path"
test ! -s "$stderr_path"

jq -e '
  .result == "PASS"
  and .tested_depths == [1,2,4,8,16,32]
  and .resident_necklace_cells == 285
  and .latent_cells_per_necklace == 2
  and .resident_joint_complex_cells == 570
  and .program_descriptor_bytes == 8
  and .retained_module_tape_bytes == 0
  and .retained_inverse_history_bytes == 0
  and .relation_table_cells == 0
  and .assignment_cells == 0
  and .temporary_occupation_cells == 0
  and .dense_285_operator_cells == 0
  and .maximum_restoration_error < 6e-11
  and .reuse_depth == 11
  and .reuse_restoration_error < 6e-11
  and .fresh_restored_reuse_boundary_error < 6e-11
  and .carrier_backing_preserved
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .controls.missing_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .controls.wrong_inverse_variant_error > 1e-5
  and .controls.wrong_owner_rejected
  and .controls.owner_attack_carrier_error == 0
  and .resource_law.maximum_native_generator_terms == 134946816
  and .resource_law.catalytic_carrier_complex_cells == 570
  and .resource_law.permanent_restoration_baseline_complex_cells == 570
  and .resource_law.per_module_fiber_complex_cells == 285
  and .resource_law.generator_matrix_complex_cells == 289
  and .resource_law.generator_plan_object_bytes == 4664
  and .resource_law.generator_plan_nonmatrix_scalar_bytes == 40
  and .resource_law.generator_work_complex_cells == 855
  and .resource_law.accepted_transaction_peak_counted_complex_cells_excluding_plan
      == 2569
  and .resource_law.verification_reference_complex_cells == 570
  and .resource_law.verification_peak_counted_complex_cells_excluding_plan
      == 3139
  and .resource_law.plan_object_bytes > 0
  and .resource_law.plan_necklace_count == 285
  and .resource_law.plan_necklace_capacity >= 285
  and .resource_law.plan_necklace_element_bytes > 0
  and .resource_law.plan_necklace_capacity_bytes
      == (.resource_law.plan_necklace_capacity
          * .resource_law.plan_necklace_element_bytes)
  and .resource_law.plan_root_complex_cells == 17
  and .resource_law.accepted_transaction_peak_counted_complex_cells_including_plan_roots
      == 2586
  and .resource_law.verification_peak_counted_complex_cells_including_plan_roots
      == 3156
  and .resource_law.native_terms_linear_in_depth
  and .resource_law.resident_carrier_constant_in_depth
  and .resource_law.inverse_descriptors_rematerialized_from_public_topology
  and (.resource_law.public_topology_inspects_final_answer | not)
  and (.resource_law.allocator_native_library_os_memory_bounded | not)
  and .strongest_compact_classical.same_570_complex_recurrence
  and .strongest_compact_classical.boundary_error == 0
  and (.machine_boundary_enforced | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$result" >/dev/null

sha256sum \
  "$source_path" \
  "$frontier_dir/four_rotor_necklace_shared_latent_owner_repair.cpp" \
  "$frontier_dir/four_rotor_necklace_shared_latent_phase.cpp" \
  "$frontier_dir/four_rotor_necklace_generator_phase.cpp" \
  "$frontier_dir/qualify_four_rotor_necklace_shared_latent_depth_compiler.sh" \
  "$binary" \
  "$result" \
  >"$evidence_dir/SHA256SUMS"

echo "four-rotor shared-latent depth compiler qualification: PASS"
