#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_cleanroom_four_rotor_necklace_bundle.sh EVIDENCE_DIR" >&2
  exit 2
fi

source_head=65be0046ae02c79ab8c3b3356ef68d891de19e53
evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$frontier_dir" rev-parse --show-toplevel)
relative_frontier=${frontier_dir#"$repo_root/"}

mkdir -p "$evidence_dir"

source_files=(
  four_rotor_rotation_quotient_phase.py
  four_rotor_streamed_momentum_coordinate_phase.py
  four_rotor_necklace_orbit_phase.cpp
  four_rotor_bosonic_givens_phase.cpp
  four_rotor_necklace_generator_phase.cpp
  four_rotor_necklace_coherence_triad.cpp
  four_rotor_necklace_relational_inference.cpp
  catvm_bosonic_givens_service_tail.inc
  catvm_bosonic_givens_protocol.py
  catvm_bosonic_givens_controller.py
  four_rotor_incremental_schmidt_closure.py
  catvm_four_rotor_incremental_backend.py
  catvm_four_rotor_incremental_service.py
)

: >"$evidence_dir/SOURCE_SHA256SUMS"
for name in "${source_files[@]}"; do
  path="$relative_frontier/$name"
  git -C "$repo_root" diff --quiet "$source_head" -- "$path"
  local_hash=$(sha256sum "$frontier_dir/$name" | cut -d ' ' -f 1)
  pinned_hash=$(git -C "$repo_root" show "$source_head:$path" | sha256sum | cut -d ' ' -f 1)
  test "$local_hash" = "$pinned_hash"
  printf '%s  %s\n' "$pinned_hash" "$path" >>"$evidence_dir/SOURCE_SHA256SUMS"
done

"$frontier_dir/qualify_four_rotor_rotation_quotient_phase.sh" \
  "$evidence_dir/rotation"
"$frontier_dir/qualify_four_rotor_streamed_momentum_coordinate_phase.sh" \
  "$evidence_dir/streamed-momentum"
"$frontier_dir/qualify_four_rotor_necklace_orbit_phase.sh" \
  "$evidence_dir/necklace"
"$frontier_dir/qualify_four_rotor_bosonic_givens_phase.sh" \
  "$evidence_dir/givens"
"$frontier_dir/qualify_four_rotor_necklace_generator_phase.sh" \
  "$evidence_dir/generator"
"$frontier_dir/qualify_four_rotor_necklace_coherence_triad.sh" \
  "$evidence_dir/coherence"
"$frontier_dir/qualify_four_rotor_necklace_relational_inference.sh" \
  "$evidence_dir/open-observation"

"$repo_root/.venv/bin/python" \
  "$frontier_dir/cleanroom_four_rotor_symmetry_oracle.py" \
  >"$evidence_dir/symmetry-oracle.json" \
  2>"$evidence_dir/symmetry-oracle.stderr"
test ! -s "$evidence_dir/symmetry-oracle.stderr"

g++ \
  -std=c++20 \
  -O2 \
  -Wall \
  -Wextra \
  -Wpedantic \
  -Werror \
  "$frontier_dir/cleanroom_four_rotor_necklace_oracle.cpp" \
  -o "$evidence_dir/cleanroom_four_rotor_necklace_oracle"
nice -n 10 taskset -c 0-3 \
  "$evidence_dir/cleanroom_four_rotor_necklace_oracle" \
  >"$evidence_dir/necklace-oracle.json" \
  2>"$evidence_dir/necklace-oracle.stderr"
test ! -s "$evidence_dir/necklace-oracle.stderr"

"$frontier_dir/qualify_catvm_bosonic_givens_atomic_repair.sh" \
  "$evidence_dir/catvm-atomic-repair"

givens_kernel=$(
  sed -n '367,406p' "$frontier_dir/four_rotor_bosonic_givens_phase.cpp"
)
if rg -q 'transition_coefficient|permanent|labelled' <<<"$givens_kernel"; then
  echo "accepted Givens kernel retained predecessor enumeration" >&2
  exit 1
fi
rg -Fq 'std::vector<Complex> polynomial(kHistogramDimension)' \
  <<<"$givens_kernel"
rg -q 'expand_necklaces' <<<"$givens_kernel"
rg -q 'close_necklaces' <<<"$givens_kernel"

generator_kernel=$(
  sed -n '136,286p' "$frontier_dir/four_rotor_necklace_generator_phase.cpp"
)
if rg -q 'kHistogramDimension|transition_coefficient|285.*285' \
    <<<"$generator_kernel"; then
  echo "generator kernel retained expanded occupation or dense operator" >&2
  exit 1
fi
rg -q 'std::vector<Complex> previous = samples' <<<"$generator_kernel"
rg -Fq 'std::vector<Complex> current(samples.size())' \
  <<<"$generator_kernel"
rg -Fq 'std::vector<Complex> next(samples.size())' \
  <<<"$generator_kernel"

jq -e '
  .result == "PASS"
  and .production_imports == 0
  and .global_rotation_quotient.sector_cells == 4913
  and .global_rotation_quotient.reduction_factor == 17
  and .necklace.weak_compositions == 4845
  and .necklace.necklaces == 285
  and .necklace.labelled_weight == 83521
  and .necklace.rotor5_necklaces == 1197
  and .necklace.exchange_symmetry_required
  and (.necklace.labelled_open_chain_compressed | not)
  and .burnside.stabilizer_correction == 1
  and .burnside.simple_division_ceiling == "R_LESS_THAN_17"
' "$evidence_dir/symmetry-oracle.json" >/dev/null

jq -e '
  .result == "PASS"
  and (.oracle_includes_production_observation_package | not)
  and (.oracle_calls_production_projection | not)
  and .carrier_cells == 285
  and .tested_programs == 5
  and .different_program_family
  and .invalid_typed_observation_rejected
  and .primary_production_parity_error < 3e-11
  and .alternate_observation_effect > 1e-5
  and .alternate_strength_effect > 1e-5
  and .reversed_module_order_effect > 1e-5
  and .different_family_effect > 1e-5
  and .maximum_restoration_error < 3e-11
  and .fresh_restored_reuse_error < 3e-11
  and .carrier_backing_preserved
  and .reuse_resource_signature_equal
  and .coherence.boundary_effect > 1e-5
  and .coherence.probability_sum_error < 3e-11
  and .coherence.transition_coefficients == 487350
  and .coherence.permanent_terms == 198838800
  and .coherence.strongest_compact_coherent_classical_error == 0
  and (.coherence.distinct_phase_resource_established | not)
  and .restoration_class == "NUMERICAL_PHYSICAL_STATE_RESTORATION"
  and .dephased_restoration_class == "SNAPSHOT_RELOAD"
  and (.shared_unresolved_observation_port_established | not)
  and (.general_inference_established | not)
' "$evidence_dir/necklace-oracle.json" >/dev/null

jq -n \
  --arg source_head "$source_head" \
  --slurpfile symmetry "$evidence_dir/symmetry-oracle.json" \
  --slurpfile oracle "$evidence_dir/necklace-oracle.json" \
  --slurpfile repair "$evidence_dir/catvm-atomic-repair/result.json" \
  '{
    schema: "four_rotor_necklace_cleanroom_bundle.v1",
    source_head: $source_head,
    copied_files: [],
    result: "PASS_WITH_DISTINCT_CATVM_REPAIR",
    verification_level: "CLEANROOM_ADVERSARIAL_VERIFICATION",
    classifications: {
      global_rotation_quotient: "INDEPENDENTLY_VERIFIED_TRANSFERABLE",
      streamed_total_momentum_coordinate: "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
      exchange_symmetric_necklace: "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
      bosonic_givens: "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
      hermitian_necklace_generator: "INDEPENDENTLY_VERIFIED_TRANSFERABLE",
      coherence_diagnostic: "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
      public_open_observation: "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
      original_staged_bosonic_catvm: "REJECTED_SOURCE_DEFECT",
      repaired_atomic_bosonic_catvm: "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
      later_incremental_catvm_ordering: "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL"
    },
    restoration: {
      quotient_necklace_givens_generator_open_observation:
        "NUMERICAL_PHYSICAL_STATE_RESTORATION",
      dephased_and_snapshot_shams: "SNAPSHOT_RELOAD",
      incremental_catvm:
        "INVERSE_PLUS_CANONICAL_NUMERICAL_QUOTIENT",
      direct_forward_only_arms: "NO_RESTORATION_CLAIM"
    },
    original_catvm_defect: {
      disconnect_after_begin_restores_borrowed_carrier: false,
      mode_separation: false,
      preserved_subclaims: [
        "ACTUAL_4845_CELL_RESIDENT_HIDDEN_OCCUPATION",
        "PROJECTION_DENIED_DURING_RESIDENCY",
        "NORMAL_PATH_FINAL_RESPONSE_AFTER_NUMERICAL_RESTORATION",
        "NORMAL_PATH_SAME_BACKING_REUSE"
      ]
    },
    repaired_catvm: $repair[0],
    symmetry_oracle: $symmetry[0],
    observation_and_coherence_oracle: $oracle[0],
    resource_law: {
      givens_resident_carrier_cells: 285,
      givens_temporary_occupation_cells: 4845,
      generator_carrier_sized_work_vectors: 3,
      generator_dense_285_operator_retained: false,
      matched_compact_classical_generator_identical: true,
      allocator_native_library_os_memory_bounded: false
    },
    rejected_interpretations: [
      "LABELLED_OPEN_CHAIN_COMPRESSED_TO_285_CELLS",
      "WHOLE_GIVENS_QUALIFICATION_PROCESS_HAS_NO_PERMANENT_COMPARISON",
      "UNQUALIFIED_TOTAL_MEMORY_ACCOUNTING",
      "MECHANICALLY_TRACKED_DIRECT_PROCESS_GENERATION_OR_LEASE",
      "SHARED_UNRESOLVED_OBSERVATION_PORT",
      "GENERAL_CATALYTIC_INFERENCE",
      "LEARNING_OR_BAYESIAN_ACCURACY",
      "DISTINCT_PHASE_RESOURCE",
      "COMPUTATIONAL_ADVANTAGE",
      "SMALL_WALL_CROSSING",
      "PHYSICAL_WAVEFORM_EXECUTION",
      "REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI"
    ],
    resume_gate: {
      compact_285_cell_carrier: true,
      generator_and_module_composition: true,
      strongest_compact_classical_baseline: true,
      restoration_classification: true,
      fresh_restored_reuse: true,
      atomic_catvm_repaired: true,
      hidden_relation_or_assignment_expansion: false,
      authority_reconciliation_pending: true,
      mechanism_gate_passed: true
    }
  }' >"$evidence_dir/bundle_result.json"

sha256sum \
  "$frontier_dir/cleanroom_four_rotor_symmetry_oracle.py" \
  "$frontier_dir/cleanroom_four_rotor_necklace_oracle.cpp" \
  "$frontier_dir/catvm_bosonic_givens_atomic_repair.cpp" \
  "$frontier_dir/catvm_bosonic_givens_atomic_repair_controller.py" \
  "$frontier_dir/qualify_catvm_bosonic_givens_atomic_repair.sh" \
  "$frontier_dir/qualify_cleanroom_four_rotor_necklace_bundle.sh" \
  "$evidence_dir/symmetry-oracle.json" \
  "$evidence_dir/necklace-oracle.json" \
  "$evidence_dir/catvm-atomic-repair/result.json" \
  "$evidence_dir/bundle_result.json" \
  >"$evidence_dir/VERIFICATION_SHA256SUMS"

echo "cleanroom four-rotor/necklace/CATVM bundle: PASS_WITH_DISTINCT_CATVM_REPAIR"
