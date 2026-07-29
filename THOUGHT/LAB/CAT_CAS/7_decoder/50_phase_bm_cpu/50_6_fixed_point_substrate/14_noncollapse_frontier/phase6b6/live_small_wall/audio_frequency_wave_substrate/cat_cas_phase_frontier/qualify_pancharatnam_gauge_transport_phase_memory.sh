#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0
ulimit -c 0

if [[ $# -ne 1 ]]; then
  echo \
    "usage: qualify_pancharatnam_gauge_transport_phase_memory.sh EVIDENCE_DIR" \
    >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/pancharatnam_gauge_transport_phase_memory.c"
oracle_path="$frontier_dir/pancharatnam_gauge_transport_oracle.py"
qualifier_path="$frontier_dir/qualify_pancharatnam_gauge_transport_phase_memory.sh"
binary="$evidence_dir/pancharatnam_gauge_transport_phase_memory"
ubsan_binary="$evidence_dir/pancharatnam_gauge_transport_phase_memory_ubsan"
result="$evidence_dir/result.json"
replay="$evidence_dir/replay.json"
ubsan_result="$evidence_dir/ubsan.json"
oracle_result="$evidence_dir/oracle.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in gcc jq cmp sha256sum nice rg; do
  command -v "$tool" >/dev/null
done
test -x "$python"

gcc \
  -std=c11 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$source_path" \
  -lm \
  -o "$binary"

gcc \
  -std=c11 \
  -O1 \
  -g \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  -fsanitize=undefined \
  -fno-sanitize-recover=all \
  "$source_path" \
  -lm \
  -o "$ubsan_binary"

nice -n 10 "$binary" >"$result" 2>"$evidence_dir/result.stderr"
nice -n 10 "$binary" >"$replay" 2>"$evidence_dir/replay.stderr"
nice -n 10 "$ubsan_binary" \
  >"$ubsan_result" 2>"$evidence_dir/ubsan.stderr"

test ! -s "$evidence_dir/result.stderr"
test ! -s "$evidence_dir/replay.stderr"
test ! -s "$evidence_dir/ubsan.stderr"
cmp "$result" "$replay"
cmp "$result" "$ubsan_result"

"$python" -m py_compile "$oracle_path"
nice -n 10 "$python" -X dev "$oracle_path" "$result" \
  >"$oracle_result" 2>"$evidence_dir/oracle.stderr"
test ! -s "$evidence_dir/oracle.stderr"

jq -e '
  .result == "PASS"
  and .claim_candidate
    == "BOUNDED_STATEFUL_PANCHARATNAM_GAUGE_TRANSPORT_ENDPOINT_INVISIBLE_HOLONOMY_PHASE_MEMORY_WITH_NUMERICAL_RESTORATION_AND_REUSE"
  and .carrier_geometry == "U1_FIBER_OVER_PUBLIC_BLOCH_SPHERE_PATH"
  and .numerical_tolerance_predeclared == 1e-10
  and .tested_segments == [4,8,16,32,64,128,256,512]
  and [.segment_runs[].segments] == [4,8,16,32,64,128,256,512]
  and ([.segment_runs[].discrete_formula_error <= 1e-10] | all)
  and ([.segment_runs[].restoration_error <= 1e-10] | all)
  and ([.segment_runs[].norm_error <= 1e-10] | all)
  and .primary.segments == 512
  and .primary.restoration_error <= 1e-10
  and .reuse.program == "PUBLIC_SPHERICAL_RECTANGLE"
  and .reuse.repetitions == 37
  and .reuse.restoration_error <= 1e-10
  and .fresh_restored_reuse_boundary_error <= 1e-10
  and .same_outer_carrier_variable
  and .actual_restored_carrier_reused
  and .restoration_generation_sequence == [1,2]
  and (.restoration_generation_lease_enforced | not)
  and .baseline_reload_bytes == 0
  and .stress_cycles == 100
  and .stress_maximum_restoration_error <= 1e-10
  and .restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and (.controls.missing_inverse_restored | not)
  and (.controls.wrong_inverse_restored | not)
  and (.controls.reordered_inverse_applicable | not)
  and .controls.premature_phase_canonicalization_erases_holonomy
  and .controls.reverse_orientation_conjugates_holonomy
  and .controls.area_perturbation_changes_holonomy
  and (.controls.null_carrier_accepted | not)
  and (.controls.endpoint_only_state_distinguishes_closed_paths | not)
  and .resource_law.phase_carrier_complex128_cells == 2
  and .resource_law.phase_carrier_inline_bytes == 32
  and .resource_law.restoration_baseline_inline_bytes == 32
  and .resource_law.final_boundary_bytes == 16
  and .resource_law.latitude_descriptor_bytes == 24
  and .resource_law.reuse_public_path_bytes == 80
  and .resource_law.retained_edge_history_bytes == 0
  and (.resource_law.whole_process_rss_claimed | not)
  and .matched_compact_classical.identical_scalar_gauge_angle_recurrence
  and .matched_compact_classical.scalar_recurrence_state_minimal
  and .matched_compact_classical.scalar_state_bytes == 8
  and .matched_compact_classical.closed_form_public_path_product_available
  and (.matched_compact_classical.runtime_advantage_claimed | not)
  and .matched_compact_classical.primary_boundary_error <= 1e-10
  and .matched_compact_classical.reuse_boundary_error <= 1e-10
  and (.intermediate_gauge_phase_projected | not)
  and .final_boundary_projected_before_inverse
  and (.catvm_custody_established | not)
  and .endpoint_invisible_scope == "PUBLIC_BLOCH_ENDPOINT_ONLY"
  and .endpoint_invisible_phase_memory_established
  and (.distinct_resource_unavailable_to_compact_classical | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.physical_waveform_execution | not)
  and (.physical_bit_replacement | not)
  and (.unbounded_computation_established | not)
  and .claim_ceiling
    == "LINUX_X86_64_DIRECT_PROCESS_COMPLEX128_TWO_CELL_PANCHARATNAM_LATITUDE_SEGMENTS4_8_16_32_64_128_256_512_PUBLIC_RECTANGLE_REUSE37_SOFTWARE_ONLY"
  and (.terminal | not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .oracle
    == "INDEPENDENT_MPMATH80_PANCHARATNAM_TRANSPORT_AND_ANALYTIC_LATITUDE_FORMULA"
  and (.production_backend_imported | not)
  and .precision_decimal_digits == 80
  and .tested_segments == [4,8,16,32,64,128,256,512]
  and [.verified_segment_runs[].segments] == [4,8,16,32,64,128,256,512]
  and .all_discrete_holonomies_match
  and .all_analytic_formulas_match
  and .all_norms_one
  and .all_restorations_below_1e_70_at_oracle_precision
  and .continuous_limit_errors_strictly_decrease
  and .primary_boundary_matches
  and .reuse_boundary_matches
  and .fresh_restored_reuse_boundary_equal
  and .reverse_orientation_conjugates_holonomy
  and .area_perturbation_changes_holonomy
  and .premature_phase_canonicalization_erases_holonomy
  and (.endpoint_only_state_distinguishes_closed_paths | not)
  and .matched_compact_classical_scalar_recurrence_identical
  and .closed_form_public_path_product_available
  and (.distinct_resource_unavailable_to_compact_classical | not)
  and (.terminal | not)
' "$oracle_result" >/dev/null

if rg -n \
  'witness_list|candidate_set|truth_table|assignment_expansion|dense_operator' \
  "$source_path" "$oracle_path"
then
  echo "Pancharatnam transport package contains forbidden extensional state" >&2
  exit 1
fi

jq -n \
  --slurpfile accepted "$result" \
  --slurpfile oracle "$oracle_result" '
  {
    result: "PASS",
    claim:
      "BOUNDED_STATEFUL_PANCHARATNAM_GAUGE_TRANSPORT_ENDPOINT_INVISIBLE_HOLONOMY_PHASE_MEMORY_WITH_NUMERICAL_RESTORATION_AND_REUSE",
    claim_ceiling:
      "LINUX_X86_64_DIRECT_PROCESS_COMPLEX128_TWO_CELL_PANCHARATNAM_LATITUDE_SEGMENTS4_8_16_32_64_128_256_512_PUBLIC_RECTANGLE_REUSE37_SOFTWARE_ONLY",
    verification_level: "INDEPENDENT_ORACLE_REEXECUTION",
    restoration_classification: "NUMERICAL_PHYSICAL_STATE_RESTORATION",
    accepted: $accepted[0],
    independent_oracle: $oracle[0],
    endpoint_invisible_phase_memory_established: true,
    catvm_custody_established: false,
    distinct_phase_resource_established: false,
    computational_advantage: false,
    small_wall_crossed: false,
    terminal: false
  }
' >"$evidence_dir/qualification.json"

hash_with_label() {
  local digest
  digest=$(sha256sum "$1")
  printf '%s  %s\n' "${digest%% *}" "$2"
}

{
  hash_with_label "$source_path" \
    "pancharatnam_gauge_transport_phase_memory.c"
  hash_with_label "$oracle_path" \
    "pancharatnam_gauge_transport_oracle.py"
  hash_with_label "$qualifier_path" \
    "qualify_pancharatnam_gauge_transport_phase_memory.sh"
  hash_with_label "$binary" \
    "pancharatnam_gauge_transport_phase_memory"
  hash_with_label "$ubsan_binary" \
    "pancharatnam_gauge_transport_phase_memory_ubsan"
  hash_with_label "$result" "result.json"
  hash_with_label "$oracle_result" "oracle.json"
  hash_with_label "$evidence_dir/qualification.json" "qualification.json"
} >"$evidence_dir/SHA256SUMS"

echo "Pancharatnam gauge-transport qualification: PASS"
