#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
  echo \
    "usage: qualify_four_rotor_necklace_cyclotomic_star_quotient.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/four_rotor_necklace_cyclotomic_star_quotient.cpp"
oracle_path="$frontier_dir/four_rotor_necklace_cyclotomic_star_quotient_oracle.py"
exact_oracle_path="$frontier_dir/four_rotor_necklace_exact_phase_precision_integer_oracle.py"
binary="$evidence_dir/four_rotor_necklace_cyclotomic_star_quotient"
ubsan_binary="$evidence_dir/four_rotor_necklace_cyclotomic_star_quotient_ubsan"
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

"$python" -m py_compile "$oracle_path" "$exact_oracle_path"
nice -n 10 taskset -c 0-3 \
  "$python" -X dev "$oracle_path" "$result" \
  >"$oracle_result" 2>"$evidence_dir/oracle.stderr"
test ! -s "$evidence_dir/oracle.stderr"

jq -e '
  .result == "PASS"
  and .claim_candidate
    == "BOUNDED_DUAL_PRIME_CONJUGATE_PAIR_CYCLOTOMIC_STAR_QUOTIENT_FIXED_570_CELL_PHASE_CLOSURE_ACROSS_DEPTH4096_WITH_EXACT_RESTORATION_AND_REUSE"
  and .quotient == "F17_X_F17_X_F41_X_F41_CONJUGATE_PAIR"
  and (.analytic_cyclotomic_amplitudes_losslessly_preserved | not)
  and .demonstrated_nonzero_kernel_element_integer == 697
  and .demonstrated_nonzero_kernel_element_maps_to_zero
  and .resident_necklace_descriptors == 285
  and .latent_cells_per_necklace == 2
  and .logical_phase_cells == 570
  and .demonstrated_nonzero_kernel_element_integer == 697
  and .demonstrated_nonzero_kernel_element_maps_to_zero
  and .residues_per_phase_cell == 4
  and .tested_depths == [1,64,256,1024,4096]
  and [.depth_runs[].forward_elementary_operations]
    == [1709,109376,437504,1750016,7000064]
  and ([.depth_runs[].forward_star_norm == [1,1,1,1]] | all)
  and ([.depth_runs[].exact_algebraic_restoration] | all)
  and ([.depth_runs[].outer_carrier_backing_preserved] | all)
  and [.exact_bridge_runs[].depth] == [1,2,4,8,16,32,64]
  and ([.exact_bridge_runs[].forward_star_norm == [1,1,1,1]] | all)
  and ([.exact_bridge_runs[].exact_algebraic_restoration] | all)
  and .primary.depth == 4096
  and .primary.forward_elementary_operations == 7000064
  and .primary.forward_star_norm == [1,1,1,1]
  and .primary.exact_algebraic_restoration
  and .reuse.depth == 1537
  and .reuse.forward_elementary_operations == 2626733
  and .reuse.forward_star_norm == [1,1,1,1]
  and .reuse.exact_algebraic_restoration
  and .fresh_reuse_boundary_equal
  and .same_outer_backing_primary_and_reuse
  and .restoration_generation_sequence == [1,2]
  and .baseline_reload_bytes == 0
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and (.controls.missing_inverse_restored | not)
  and (.controls.wrong_inverse_variant_restored | not)
  and (.controls.reordered_inverse_restored | not)
  and .controls.phase_disabled_boundary_differs
  and .controls.topology_perturbation_boundary_differs
  and .controls.null_carrier_boundary_differs
  and (.controls.wrong_conjugation_pairing_norm_is_one | not)
  and .resource_law.logical_phase_cells_constant
  and .resource_law.residues_per_phase_cell_constant
  and .resource_law.phase_cell_bytes == 8
  and .resource_law.carrier_vector_capacity_cells == 570
  and .resource_law.carrier_inline_bytes == 4560
  and .resource_law.restoration_baseline_inline_bytes == 4560
  and .resource_law.permanent_restoration_baseline_phase_cells == 570
  and .resource_law.final_boundary_phase_cells == 7
  and .resource_law.topology_necklace_count == 285
  and .resource_law.topology_necklace_capacity == 285
  and .resource_law.topology_necklace_element_bytes == 36
  and .resource_law.topology_necklace_capacity_bytes == 10260
  and .resource_law.transaction_run_record_bytes == 112
  and .resource_law.accepted_path_declared_payload_subtotal_bytes == 19492
  and .resource_law.primary_forward_elementary_operations == 7000064
  and .resource_law.primary_forward_inverse_elementary_operations
    == 14000128
  and .resource_law.reuse_forward_elementary_operations == 2626733
  and .resource_law.reuse_forward_inverse_elementary_operations == 5253466
  and .resource_law.retained_module_tape_bytes == 0
  and .resource_law.retained_inverse_history_bytes == 0
  and (.resource_law.public_topology_inspects_final_answer | not)
  and .resource_law.dense_570_by_570_operator_cells == 0
  and .resource_law.relation_table_cells == 0
  and .resource_law.assignment_cells == 0
  and (.resource_law.allocator_control_block_and_heap_metadata_bounded | not)
  and (.resource_law.verification_harness_and_binary_bytes_included | not)
  and (.resource_law.whole_process_rss_claimed | not)
  and .strongest_compact_classical.identical_four_residue_recurrence
  and .strongest_compact_classical.boundary_error == 0
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
    == "INDEPENDENT_CONJUGATE_PAIR_RESIDUE_RECURRENCE_WITH_FRACTION_EXACT_BRIDGE"
  and .histogram_count == 4845
  and .necklace_count == 285
  and .logical_phase_cells == 570
  and [.embeddings[].prime] == [17,17,41,41]
  and [.embeddings[].root] == [2,9,3,14]
  and [.embeddings[].conjugate_index] == [1,0,3,2]
  and .tested_depths == [1,64,256,1024,4096]
  and .exact_bridge_depths == [1,2,4,8,16,32,64]
  and .fresh_restored_reuse_boundary_equal
  and .all_residue_boundaries_match
  and .all_star_norms_one
  and .all_restorations_exact
  and .all_exact_fraction_bridges_match
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
  echo "cyclotomic star quotient contains forbidden extensional state" >&2
  exit 1
fi

jq -n \
  --slurpfile accepted "$result" \
  --slurpfile oracle "$oracle_result" '
  {
    result: "PASS",
    claim:
      "BOUNDED_DUAL_PRIME_CONJUGATE_PAIR_CYCLOTOMIC_STAR_QUOTIENT_FIXED_570_CELL_PHASE_CLOSURE_ACROSS_DEPTH4096_WITH_EXACT_RESTORATION_AND_REUSE",
    claim_ceiling:
      "DIRECT_PROCESS_GRID17_FOUR_EXCHANGE_SYMMETRIC_ROTATION_INVARIANT_ROTORS_285_NECKLACE_DESCRIPTORS_570_F17_F17_F41_F41_CONJUGATE_PAIR_RESIDUE_PHASE_CELLS_PUBLIC_VARIANT_ORDINAL_MATCHING_COMPILER_DEPTHS1_64_256_1024_4096_REUSE_DEPTH1537_SOFTWARE_ONLY",
    verification_level: "INDEPENDENT_ORACLE_REEXECUTION",
    restoration_classification: "EXACT_ALGEBRAIC_RESTORATION",
    accepted: $accepted[0],
    independent_oracle: $oracle[0],
    fixed_width_exact_quotient_semantics: true,
    analytic_cyclotomic_amplitudes_losslessly_preserved: false,
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
  "$exact_oracle_path" \
  "$frontier_dir/qualify_four_rotor_necklace_cyclotomic_star_quotient.sh" \
  "$binary" \
  "$ubsan_binary" \
  "$result" \
  "$oracle_result" \
  "$evidence_dir/qualification.json" \
  >"$evidence_dir/SHA256SUMS"

echo "four-rotor necklace cyclotomic star quotient qualification: PASS"
