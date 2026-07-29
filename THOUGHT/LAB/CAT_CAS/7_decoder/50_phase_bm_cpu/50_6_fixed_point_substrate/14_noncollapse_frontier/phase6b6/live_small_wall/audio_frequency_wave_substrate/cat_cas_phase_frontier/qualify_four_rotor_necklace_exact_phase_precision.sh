#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
  echo \
    "usage: qualify_four_rotor_necklace_exact_phase_precision.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/four_rotor_necklace_exact_phase_precision.cpp"
oracle_path="$frontier_dir/four_rotor_necklace_exact_phase_precision_oracle.py"
binary="$evidence_dir/four_rotor_necklace_exact_phase_precision"
ubsan_binary="$evidence_dir/four_rotor_necklace_exact_phase_precision_ubsan"
result="$evidence_dir/result.json"
replay="$evidence_dir/replay.json"
ubsan_result="$evidence_dir/ubsan.json"
oracle_result="$evidence_dir/oracle.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in g++ jq cmp sha256sum nice taskset; do
  command -v "$tool" >/dev/null
done
test -x "$python"

g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$source_path" \
  -o "$binary"

g++ \
  -std=c++20 \
  -O1 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  -fsanitize=undefined \
  -fno-sanitize-recover=all \
  "$source_path" \
  -o "$ubsan_binary"

nice -n 10 taskset -c 0-3 "$binary" \
  >"$result" 2>"$evidence_dir/result.stderr"
nice -n 10 taskset -c 0-3 "$binary" \
  >"$replay" 2>"$evidence_dir/replay.stderr"
nice -n 10 taskset -c 0-3 "$ubsan_binary" \
  >"$ubsan_result" 2>"$evidence_dir/ubsan.stderr"

test ! -s "$evidence_dir/result.stderr"
test ! -s "$evidence_dir/replay.stderr"
test ! -s "$evidence_dir/ubsan.stderr"
cmp "$result" "$replay"
cmp "$result" "$ubsan_result"

"$python" -m py_compile "$oracle_path"
nice -n 10 taskset -c 0-3 \
  "$python" -X dev "$oracle_path" "$result" \
  >"$oracle_result" 2>"$evidence_dir/oracle.stderr"
test ! -s "$evidence_dir/oracle.stderr"

jq -e '
  .result == "PASS"
  and .claim_candidate
    == "BOUNDED_EXACT_DYADIC_CYCLOTOMIC_PHASE_PRECISION_GROWTH_ON_FIXED_570_CELL_NECKLACE_LATENT_CARRIER_WITH_EXACT_RESTORATION_AND_REUSE"
  and .field == "Z[ZETA8,1/2]"
  and .resident_necklace_descriptors == 285
  and .latent_cells_per_necklace == 2
  and .logical_phase_cells == 570
  and .arbitrary_precision_integer_slots == 2280
  and .tested_depths == [1,2,4,8,16,32,64]
  and [.depth_runs[].forward_elementary_operations]
    == [1709,3418,6836,13672,27344,54688,109376]
  and [.depth_runs[].maximum_numerator_bits]
    == [1,1,1,4,7,15,31]
  and [.depth_runs[].maximum_denominator_power]
    == [1,2,3,5,9,17,33]
  and [.depth_runs[].forward_logical_payload_bits]
    == [2854,2864,2898,3188,4339,14645,57166]
  and ([.depth_runs[].exact_local_unitaries] | all)
  and ([.depth_runs[].exact_algebraic_restoration] | all)
  and ([.depth_runs[].carrier_backing_preserved] | all)
  and .primary.depth == 64
  and .primary.maximum_numerator_bits == 31
  and .primary.maximum_denominator_power == 33
  and .primary.forward_logical_payload_bits == 57166
  and .reuse.depth == 23
  and .reuse.maximum_numerator_bits == 10
  and .reuse.maximum_denominator_power == 12
  and .reuse.forward_logical_payload_bits == 7880
  and .fresh_reuse_boundary_equal
  and .same_backing_primary_and_reuse
  and .carrier_backing_identity_scope
    == "OUTER_570_PHASE_OBJECT_VECTOR_BIG_INTEGER_LIMB_ALLOCATIONS_NOT_STABLE_OR_CLAIMED"
  and .restoration_generation_sequence == [1,2]
  and .baseline_reload_bytes == 0
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and (.controls.missing_inverse_restored | not)
  and (.controls.wrong_inverse_variant_restored | not)
  and (.controls.reordered_inverse_restored | not)
  and .controls.phase_disabled_boundary_differs
  and .controls.topology_perturbation_boundary_differs
  and .controls.null_carrier_boundary_differs
  and .resource_law.logical_phase_cells_constant
  and .resource_law.baseline_logical_payload_bits == 2851
  and .resource_law.phase_object_bytes > 0
  and .resource_law.carrier_vector_capacity_cells == 570
  and .resource_law.carrier_inline_object_bytes
    == (570 * .resource_law.phase_object_bytes)
  and .resource_law.permanent_restoration_baseline_phase_cells == 570
  and .resource_law.final_boundary_phase_cells == 7
  and .resource_law.topology_necklace_count == 285
  and .resource_law.topology_necklace_capacity == 285
  and .resource_law.topology_necklace_element_bytes > 0
  and .resource_law.topology_necklace_capacity_bytes
    == (.resource_law.topology_necklace_capacity
      * .resource_law.topology_necklace_element_bytes)
  and .resource_law.accepted_primary_forward_elementary_operations
    == 109376
  and .resource_law.accepted_primary_forward_inverse_elementary_operations
    == 218752
  and .resource_law.accepted_reuse_forward_elementary_operations
    == 39307
  and .resource_law.accepted_reuse_forward_inverse_elementary_operations
    == 78614
  and .resource_law.denominator_power_strictly_grows_on_tested_depths
  and .resource_law.logical_payload_strictly_grows_on_tested_depths
  and .resource_law.retained_module_tape_bytes == 0
  and .resource_law.retained_inverse_history_bytes == 0
  and (.resource_law.public_topology_inspects_final_answer | not)
  and .resource_law.dense_570_by_570_operator_cells == 0
  and .resource_law.relation_table_cells == 0
  and .resource_law.assignment_cells == 0
  and (.resource_law.whole_process_rss_claimed | not)
  and (.resource_law.big_integer_allocator_and_container_overhead_bounded | not)
  and (.resource_law.temporary_object_lifetime_peak_bounded | not)
  and .strongest_compact_classical.identical_570_cell_exact_recurrence
  and .strongest_compact_classical.boundary_residue_error == 0
  and .distinct_from_complex128_hermitian_generator_package
  and (.intermediate_projected | not)
  and (.machine_boundary_enforced | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.physical_bit_replacement | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .oracle
    == "INDEPENDENT_DUAL_FINITE_FIELD_PUBLIC_TOPOLOGY_RECURRENCE"
  and .histogram_count == 4845
  and .necklace_count == 285
  and .logical_phase_cells == 570
  and .tested_depths == [1,2,4,8,16,32,64]
  and [.fields[].prime] == [17,41]
  and [.fields[].primitive_eighth_root] == [2,3]
  and ([.fields[].all_boundaries_match] | all)
  and ([.fields[].all_restorations_exact] | all)
  and ([.fields[].all_forward_norms_one] | all)
  and (.production_backend_imported | not)
  and .dense_operator_cells == 0
  and .assignment_expansion_cells == 0
  and .matched_compact_classical_recurrence_identical
  and (.terminal | not)
' "$oracle_result" >/dev/null

if rg -n \
  'witness_list|candidate_set|truth_table|dense_operator\[' \
  "$source_path" "$oracle_path"
then
  echo "exact precision path contains forbidden extensional state" >&2
  exit 1
fi

jq -n \
  --slurpfile accepted "$result" \
  --slurpfile oracle "$oracle_result" '
  {
    result: "PASS",
    claim:
      "BOUNDED_EXACT_DYADIC_CYCLOTOMIC_PHASE_PRECISION_GROWTH_ON_FIXED_570_CELL_NECKLACE_LATENT_CARRIER_WITH_EXACT_RESTORATION_AND_REUSE",
    claim_ceiling:
      "DIRECT_PROCESS_GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACE_DESCRIPTORS_570_Q_ZETA8_DYADIC_PHASE_CELLS_PUBLIC_VARIANT_ORDINAL_MATCHING_COMPILER_DEPTHS1_2_4_8_16_32_64_REUSE_DEPTH23_SOFTWARE_ONLY",
    verification_level: "INDEPENDENT_ORACLE_REEXECUTION",
    restoration_classification: "EXACT_ALGEBRAIC_RESTORATION",
    accepted: $accepted[0],
    independent_oracle: $oracle[0],
    fixed_logical_cells_hide_growing_exact_coefficient_width: true,
    strongest_compact_classical_recurrence_identical: true,
    distinct_phase_resource_established: false,
    computational_advantage: false,
    small_wall_crossed: false,
    terminal: false
  }
' >"$evidence_dir/qualification.json"

sha256sum \
  "$source_path" \
  "$oracle_path" \
  "$frontier_dir/qualify_four_rotor_necklace_exact_phase_precision.sh" \
  "$binary" \
  "$ubsan_binary" \
  "$result" \
  "$oracle_result" \
  "$evidence_dir/qualification.json" \
  >"$evidence_dir/SHA256SUMS"

echo "four-rotor necklace exact phase precision qualification: PASS"
