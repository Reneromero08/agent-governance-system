#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_four_rotor_necklace_generator_phase.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_file="$frontier_dir/four_rotor_necklace_generator_phase.cpp"
bosonic_source="$frontier_dir/four_rotor_bosonic_givens_phase.cpp"
necklace_source="$frontier_dir/four_rotor_necklace_orbit_phase.cpp"
binary="$evidence_dir/four_rotor_necklace_generator_phase"
result="$evidence_dir/result.json"
stderr_file="$evidence_dir/stderr.txt"

mkdir -p "$evidence_dir"
g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$source_file" \
  -o "$binary"

nice -n 10 taskset -c 0-3 "$binary" \
  >"$result" 2>"$stderr_file"

jq -e '
  .result == "PASS"
  and .resident_necklace_complex_cells == 285
  and .accepted_path_temporary_necklace_work_complex_cells == 855
  and .accepted_path_temporary_occupation_complex_cells == 0
  and .comparison_path_temporary_occupation_complex_cells == 4845
  and .comparison_bosonic_givens_occupation_scratch_bytes == 77520
  and .accepted_path_retained_transition_operator_bytes == 0
  and .accepted_path_permanent_assignment_terms_enumerated == 0
  and (.accepted_path_occupation_expansion_materialized | not)
  and .comparison_path_occupation_expansion_materialized
  and .parity.one_step_bosonic_givens_l2_error < 3e-11
  and .parity.depth8_bosonic_givens_boundary_error < 3e-11
  and .parity.maximum_eigenvalue_modulus_error < 3e-13
  and .parity.maximum_generator_hermitian_error < 3e-13
  and .parity.maximum_chebyshev_tail_bound < 1e-13
  and .primary.weighted_norm_error < 3e-11
  and .primary.restoration_error < 3e-11
  and .primary.actual_inverse_restoration
  and .primary.resources.carrier_payload_bytes == 4560
  and .primary.resources.occupation_scratch_bytes == 0
  and .primary.resources.maximum_explicit_engine_bytes == 33470
  and .primary.resources.maximum_explicit_wrapper_bytes == 38142
  and .primary.resources.retained_inverse_history_bytes == 0
  and .reuse.restoration_generation == 2
  and .reuse.restoration_error < 3e-11
  and .reuse.fresh_restored_boundary_error < 3e-11
  and .reuse.actual_restored_carrier_reuse
  and .controls.missing_inverse_error > 1e-5
  and .controls.wrong_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .matched_classical_generator_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$result" >/dev/null

test ! -s "$stderr_file"

sha256sum \
  "$source_file" \
  "$bosonic_source" \
  "$necklace_source" \
  "$frontier_dir/qualify_four_rotor_necklace_generator_phase.sh" \
  "$binary" \
  "$result" \
  >"$evidence_dir/SHA256SUMS"

echo "four-rotor necklace generator qualification: PASS"
