#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_four_rotor_bosonic_givens_phase.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_file="$frontier_dir/four_rotor_bosonic_givens_phase.cpp"
dependency="$frontier_dir/four_rotor_necklace_orbit_phase.cpp"
mkdir -p "$evidence_dir"
binary="$evidence_dir/four_rotor_bosonic_givens_phase"
result="$evidence_dir/result.json"
stderr_file="$evidence_dir/stderr"

g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$source_file" \
  -o "$binary"

"$binary" >"$result" 2>"$stderr_file"

jq -e '
  .result == "PASS"
  and .resident_necklace_complex_cells == 285
  and .temporary_occupation_complex_cells == 4845
  and .labelled_wave_cells_avoided == 83521
  and .retained_transition_operator_bytes == 0
  and .accepted_path_permanent_assignment_terms_enumerated == 0
  and .comparison_predecessor_permanent_assignment_terms_enumerated == 530236800
  and .parity.one_step_predecessor_l2_error < 3e-11
  and .parity.depth8_predecessor_boundary_error < 3e-11
  and .parity.maximum_single_particle_decomposition_error < 3e-13
  and .parity.maximum_necklace_closure_error < 3e-11
  and .primary.depth == 8
  and .primary.weighted_norm_error < 3e-11
  and .primary.restoration_error < 3e-11
  and .primary.actual_inverse_restoration
  and .primary.warm_elapsed_reduction_factor > 1
  and .primary.enumerated_term_count_ratio > 10
  and .primary.predecessor_permanent_terms == 530236800
  and .primary.accepted_polynomial_block_terms == 15917440
  and .primary.resources.carrier_payload_bytes == 4560
  and .primary.resources.occupation_scratch_bytes == 77520
  and .primary.resources.polynomial_block_scratch_bytes == 211
  and .primary.resources.maximum_explicit_engine_bytes == 97447
  and .primary.resources.maximum_explicit_wrapper_bytes == 102119
  and .primary.resources.comparison_harness_peak_explicit_bytes == 106567
  and .primary.resources.retained_inverse_history_bytes == 0
  and .primary.resources.occupation_expansion_materialized
  and (.primary.resources.labelled_assignment_expansion_materialized | not)
  and .primary.resources.givens_two_mode_updates > 0
  and .primary.resources.polynomial_block_terms > 0
  and .reuse.restoration_generation == 2
  and .reuse.restoration_error < 3e-11
  and .reuse.fresh_restored_boundary_error < 3e-11
  and .reuse.actual_restored_carrier_reuse
  and .controls.missing_inverse_error > 1e-5
  and .controls.wrong_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .matched_classical_bosonic_givens_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$result" >/dev/null

test ! -s "$stderr_file"

sha256sum \
  "$source_file" \
  "$dependency" \
  "$frontier_dir/qualify_four_rotor_bosonic_givens_phase.sh" \
  "$result" \
  >"$evidence_dir/SHA256SUMS"

echo "four-rotor bosonic Givens phase qualification: PASS"
