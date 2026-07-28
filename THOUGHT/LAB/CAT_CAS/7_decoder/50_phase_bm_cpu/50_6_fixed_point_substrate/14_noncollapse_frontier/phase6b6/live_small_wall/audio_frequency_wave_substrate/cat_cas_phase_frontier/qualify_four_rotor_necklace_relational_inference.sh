#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_four_rotor_necklace_relational_inference.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
source_file="$frontier_dir/four_rotor_necklace_relational_inference.cpp"
triad_source="$frontier_dir/four_rotor_necklace_coherence_triad.cpp"
generator_source="$frontier_dir/four_rotor_necklace_generator_phase.cpp"
bosonic_source="$frontier_dir/four_rotor_bosonic_givens_phase.cpp"
necklace_source="$frontier_dir/four_rotor_necklace_orbit_phase.cpp"
binary="$evidence_dir/four_rotor_necklace_relational_inference"
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
  and .typed_open_observation_relations.ports == 6
  and (.typed_open_observation_relations.kinds | index("COLLISION")) != null
  and (.typed_open_observation_relations.kinds | index("CYCLIC_SEPARATION")) != null
  and .typed_open_observation_relations.cyclic_separations == [1,3,5,8]
  and .typed_open_observation_relations.relation == "R(x,o)=OMEGA^(STRENGTH*(FEATURE(x)-o)^2)"
  and .typed_open_observation_relations.observation_domains.COLLISION == [0,6]
  and .typed_open_observation_relations.observation_domains.CYCLIC_SEPARATION == [0,4]
  and .typed_open_observation_relations.public_observations == [1,2,0,3,1,4]
  and .typed_open_observation_relations.native_substitution_closure
  and (.typed_open_observation_relations.observation_domain_enumerated | not)
  and .typed_open_observation_relations.relation_table_cells_materialized == 0
  and .typed_open_observation_relations.intermediate_necklace_cells == 285
  and (.typed_open_observation_relations.accepted_path_intermediate_projected | not)
  and .typed_open_observation_relations.accepted_path_truth_table_cells == 0
  and .typed_open_observation_relations.accepted_path_candidate_assignment_cells == 0
  and .inference_boundary.hypothesis_variable == "TOTAL_UNORDERED_OCCUPATION_COLLISION_COUNT"
  and .inference_boundary.hypothesis_domain == [0,1,2,3,4,5,6]
  and .inference_boundary.observation_semantics == "PUBLIC_EXACT_ROTATION_INVARIANT_FEATURE_VALUES_CLOSED_INTO_TYPED_PHASE_RELATIONS"
  and .inference_boundary.score_semantics == "WEIGHTED_COLLISION_HYPOTHESIS_INTERFERENCE_SCORE_NOT_CALIBRATED_BAYESIAN_POSTERIOR"
  and .inference_boundary.decision_rule == "ARGMAX_SCORE_LOWEST_INDEX_TIEBREAK"
  and (.inference_boundary.selected_hypothesis >= 0)
  and (.inference_boundary.selected_hypothesis <= 6)
  and .inference_boundary.independent_semantic_reference_error < 3e-11
  and .inference_boundary.retained_outside_inverse_history
  and .primary.restoration_generation == 1
  and .primary.restoration_error < 3e-11
  and .primary.weighted_norm_error < 3e-11
  and .primary.actual_inverse_restoration
  and .primary.carrier_backing_preserved
  and .primary.evidence_phase_updates == 3420
  and .primary.open_observation_ports_closed == 12
  and .primary.relation_table_cells_materialized == 0
  and .primary.generator_applications == 768
  and .primary.streamed_generator_terms == 12651264
  and .primary.engine_explicit_payload_bytes == 33638
  and .primary.verification_baseline_bytes == 4560
  and .primary.projection_boundary_bytes == 56
  and .primary.lifecycle_explicit_payload_bytes == 38310
  and .reuse.restoration_generation == 2
  and .reuse.restoration_error < 3e-11
  and .reuse.fresh_restored_boundary_error < 3e-11
  and .reuse.actual_restored_carrier_reuse
  and .reuse.carrier_backing_preserved
  and .causal_controls.bypassed_evidence_boundary_effect > 1e-5
  and .causal_controls.reordered_module_boundary_effect > 1e-5
  and .causal_controls.initial_and_interstep_dephasing_boundary_effect > 1e-5
  and .causal_controls.missing_inverse_error > 1e-5
  and .causal_controls.wrong_inverse_error > 1e-5
  and .causal_controls.reordered_inverse_error > 1e-5
  and .dephased_comparison.initial_and_each_step_necklace_basis_dephasing
  and (.dephased_comparison.evidence_phase_contribution_isolated | not)
  and .dephased_comparison.outside_accepted_coherent_path
  and .dephased_comparison.streamed_transition_coefficients == 487350
  and .dephased_comparison.permanent_assignment_terms == 198838800
  and .matched_compact_classical.same_executable_complex_recurrence
  and .matched_compact_classical.boundary_error == 0
  and .matched_compact_classical.restoration_error < 3e-11
  and .topology_compilation_weak_compositions == 4845
  and (.machine_boundary_enforced | not)
  and (.no_smuggle_enforced | not)
  and (.accepted_path_intermediate_emitted | not)
  and .bounded_catalytic_hypothesis_scoring_contract_established
  and (.general_catalytic_inference_established | not)
  and (.ground_truth_accuracy_established | not)
  and (.calibrated_statistical_inference_established | not)
  and (.learning_advantage_established | not)
  and (.distinct_phase_resource_established | not)
  and (.computational_advantage | not)
  and (.small_wall_crossed | not)
  and (.unbounded_computation_established | not)
  and (.terminal | not)
' "$result" >/dev/null

test ! -s "$stderr_file"

sha256sum \
  "$source_file" \
  "$triad_source" \
  "$generator_source" \
  "$bosonic_source" \
  "$necklace_source" \
  "$frontier_dir/qualify_four_rotor_necklace_relational_inference.sh" \
  "$binary" \
  "$result" \
  >"$evidence_dir/SHA256SUMS"

echo "four-rotor necklace relational inference qualification: PASS"
