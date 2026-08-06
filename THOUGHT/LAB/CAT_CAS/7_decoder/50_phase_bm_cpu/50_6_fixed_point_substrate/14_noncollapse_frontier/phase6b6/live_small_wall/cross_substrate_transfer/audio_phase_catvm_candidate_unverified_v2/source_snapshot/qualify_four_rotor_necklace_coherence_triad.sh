#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_four_rotor_necklace_coherence_triad.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_file="$frontier_dir/four_rotor_necklace_coherence_triad.cpp"
generator_source="$frontier_dir/four_rotor_necklace_generator_phase.cpp"
bosonic_source="$frontier_dir/four_rotor_bosonic_givens_phase.cpp"
necklace_source="$frontier_dir/four_rotor_necklace_orbit_phase.cpp"
binary="$evidence_dir/four_rotor_necklace_coherence_triad"
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
  and .public_instance.grid == 17
  and .public_instance.rotors == 4
  and .public_instance.depth == 8
  and .public_instance.resident_necklace_cells == 285
  and .coherent_in_place.restoration_error < 3e-11
  and .coherent_in_place.reuse_restoration_error < 3e-11
  and .coherent_in_place.actual_inverse_restoration
  and .coherent_in_place.actual_restored_carrier_reuse
  and .coherent_in_place.timing_scope == "FORWARD_PROJECTION_ACTUAL_INVERSE"
  and .coherent_in_place.engine_explicit_payload_bytes == 33470
  and .coherent_in_place.verification_baseline_bytes == 4560
  and .coherent_in_place.projection_boundary_bytes == 56
  and .coherent_in_place.lifecycle_explicit_payload_bytes == 38142
  and .dephased_snapshot_sham.basis == "GLOBAL_ROTATION_NECKLACE_ORBIT"
  and .dephased_snapshot_sham.initial_carrier_dephased
  and .dephased_snapshot_sham.dephased_after_each_free_step
  and .dephased_snapshot_sham.coherence_boundary_effect > 1e-5
  and .dephased_snapshot_sham.maximum_probability_sum_error < 3e-11
  and .dephased_snapshot_sham.collision_phase_updates_erased == 2280
  and .dephased_snapshot_sham.streamed_transition_coefficients == 649800
  and .dephased_snapshot_sham.permanent_assignment_terms == 265118400
  and .dephased_snapshot_sham.probability_payload_bytes == 4560
  and .dephased_snapshot_sham.public_topology_bytes == 10532
  and .dephased_snapshot_sham.transition_scratch_bytes == 254
  and .dephased_snapshot_sham.timing_scope == "FORWARD_ONLY_EXCLUDES_SNAPSHOT_AND_REUSE"
  and (.dephased_snapshot_sham.lifecycle_timing_measured | not)
  and .dephased_snapshot_sham.forward_explicit_payload_bytes == 24522
  and .dephased_snapshot_sham.lifecycle_explicit_payload_bytes == 42702
  and .dephased_snapshot_sham.snapshot_creation_bytes == 4560
  and .dephased_snapshot_sham.snapshot_reload_bytes == 4560
  and (.dephased_snapshot_sham.actual_inverse_restoration | not)
  and .dephased_snapshot_sham.snapshot_backed_reuse_only
  and .dephased_snapshot_sham.reuse_restoration_error_after_reload < 3e-11
  and .matched_compact_classical.same_executable_recurrence
  and .matched_compact_classical.boundary_error == 0
  and .matched_compact_classical.reuse_boundary_error < 3e-11
  and .matched_compact_classical.restoration_error < 3e-11
  and .matched_compact_classical.reuse_restoration_error < 3e-11
  and .matched_compact_classical.actual_inverse_restoration
  and .matched_compact_classical.actual_restored_carrier_reuse
  and .matched_compact_classical.timing_scope == "FORWARD_PROJECTION_ACTUAL_INVERSE"
  and .matched_compact_classical.engine_explicit_payload_bytes == 33470
  and .matched_compact_classical.verification_baseline_bytes == 4560
  and .matched_compact_classical.projection_boundary_bytes == 56
  and .matched_compact_classical.lifecycle_explicit_payload_bytes == 38142
  and (.traffic.protocol_used | not)
  and .traffic.logical_protocol_bytes_per_arm == 0
  and .comparison_harness.conservative_explicit_payload_bytes == 66894
  and .controls.missing_inverse_error > 1e-5
  and .controls.wrong_inverse_error > 1e-5
  and .controls.reordered_inverse_error > 1e-5
  and .diagnosis.initial_and_interstep_necklace_coherence_changes_boundary
  and (.diagnosis.collision_phase_contribution_isolated | not)
  and .diagnosis.matched_classical_inherits_coherence_exactly
  and (.diagnosis.distinct_phase_resource_established | not)
  and (.diagnosis.computational_advantage | not)
  and (.diagnosis.small_wall_crossed | not)
  and (.diagnosis.unbounded_computation_established | not)
  and (.terminal | not)
' "$result" >/dev/null

test ! -s "$stderr_file"

sha256sum \
  "$source_file" \
  "$generator_source" \
  "$bosonic_source" \
  "$necklace_source" \
  "$frontier_dir/qualify_four_rotor_necklace_coherence_triad.sh" \
  "$binary" \
  "$result" \
  >"$evidence_dir/SHA256SUMS"

echo "four-rotor necklace coherence triad qualification: PASS"
