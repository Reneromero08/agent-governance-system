#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_four_rotor_necklace_orbit_phase.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_file="$frontier_dir/four_rotor_necklace_orbit_phase.cpp"
mkdir -p "$evidence_dir"

binary="$evidence_dir/four_rotor_necklace_orbit_phase"
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
  and .grid_size == 17
  and .rotors_executed == 4
  and .labelled_rotation_quotient_cells == 4913
  and .necklace_carrier_complex_cells == 285
  and .four_rotor_state_reduction_factor > 17
  and .dimension_law.histograms_r4 == 4845
  and .dimension_law.necklaces_r4 == 285
  and .dimension_law.necklaces_r5_analytic == 1197
  and .symmetry_gates.particle_permutation_commutes
  and .symmetry_gates.global_rotation_commutes
  and .symmetry_gates.nonseparable_collision_phase
  and .symmetry_gates.circulant_quadratic_free_phase
  and .primary.depth == 8
  and .primary.weighted_norm_error < 2e-11
  and .primary.restoration_error < 3e-11
  and .primary.actual_inverse_restoration
  and .primary.resources.carrier_payload_bytes == 4560
  and .primary.resources.plan_compilation_conservative_explicit_payload_bytes == 10619
  and .primary.resources.transition_scratch_bytes == 177
  and .primary.resources.retained_inverse_history_bytes == 0
  and .primary.resources.retained_transition_operator_bytes == 0
  and (.primary.resources.labelled_wave_materialized_in_accepted_path | not)
  and (.primary.resources.assignment_expansion_materialized_in_accepted_path | not)
  and .primary.resources.stored_assignment_list_bytes == 0
  and .primary.resources.permanent_assignment_terms_enumerated > 0
  and .primary.resources.streamed_transition_coefficients > 0
  and .primary.resources.exact_cyclotomic_permanent_terms > 0
  and (.verification.labelled_wave_used_for_restoration | not)
  and .verification.labelled_wave_embedding_l2_error < 2e-11
  and .reuse.restoration_generation == 2
  and .reuse.restoration_error < 3e-11
  and .reuse.fresh_restored_boundary_error < 2e-11
  and .reuse.actual_restored_carrier_reuse
  and .controls.missing_inverse_error > 1e-5
  and .controls.wrong_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .phase_primitive_state == "UNRESOLVED_CYCLOTOMIC_NECKLACE_AMPLITUDES"
  and (.intermediate_projected | not)
  and .matched_classical_orbit_simulator_identical
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.original_open_chain_program_family_compressed | not)
  and (.terminal | not)
' "$result" >/dev/null

test ! -s "$stderr_file"

sha256sum \
  "$source_file" \
  "$frontier_dir/qualify_four_rotor_necklace_orbit_phase.sh" \
  "$result" \
  >"$evidence_dir/SHA256SUMS"

echo "four-rotor necklace-orbit phase qualification: PASS"
