#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_runtime_weighted_grid_kronecker_butterfly_closure.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_runtime_weighted_grid_kronecker_butterfly_closure.py"
oracle_path="$frontier_dir/f17_runtime_weighted_grid_kronecker_butterfly_closure_oracle.py"
benchmark_path="$frontier_dir/benchmark_f17_runtime_weighted_grid_kronecker_butterfly_closure.py"
m119_path="$frontier_dir/f17_runtime_weighted_grid_phase_factor_closure.py"
m119_expected_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_PHASE_FACTOR_CLOSURE_RESULTS.json"
m118_path="$frontier_dir/f17_variable_rank_nonseparable_tensor_coupling.py"
m116_path="$frontier_dir/f17_cubic_chain_period17_quadratic_extension_resident_carrier.py"
expected_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_KRONECKER_BUTTERFLY_CLOSURE_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_KRONECKER_BUTTERFLY_CLOSURE_ORACLE_RESULTS.json"
benchmark_expected_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_KRONECKER_BUTTERFLY_CLOSURE_BENCHMARK.json"
provenance_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_KRONECKER_BUTTERFLY_CLOSURE_PROVENANCE.json"
review_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_KRONECKER_BUTTERFLY_CLOSURE_INDEPENDENT_REVIEW.md"
qualifier_path="$frontier_dir/qualify_f17_runtime_weighted_grid_kronecker_butterfly_closure.sh"
result="$evidence_dir/m120.qualifier.result.full.json"
oracle_result="$evidence_dir/m120.qualifier.oracle.full.json"
benchmark_result="$evidence_dir/m120.qualifier.benchmark.observed.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in cmp git jq nice rg sha256sum; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path" "$benchmark_expected_path" "$provenance_path"

scientific_parent=$(jq -r '.scientific_source_parent' "$provenance_path")
test "$scientific_parent" = "a87d1c2c5e5bec30e26ae39512b17313d0cec8b9"
git -C "$repo" cat-file -e "$scientific_parent^{commit}"
git -C "$repo" merge-base --is-ancestor "$scientific_parent" HEAD

for sealed_path in \
  "$source_path" \
  "$oracle_path" \
  "$benchmark_path" \
  "$m119_path" \
  "$m119_expected_path" \
  "$m118_path" \
  "$m116_path" \
  "$qualifier_path" \
  "$expected_path" \
  "$oracle_expected_path" \
  "$benchmark_expected_path" \
  "$review_path"
do
  sealed_name=$(basename "$sealed_path")
  sealed_expected=$(jq -r --arg name "$sealed_name" '.files[$name] // empty' "$provenance_path")
  test -n "$sealed_expected"
  test "$(sha256sum "$sealed_path" | cut -d' ' -f1)" = "$sealed_expected"
done

if rg -n '^[[:space:]]*(from|import)[[:space:]]+(f17_runtime_weighted_grid_kronecker_butterfly_closure|f17_runtime_weighted_grid_phase_factor_closure|f17_variable_rank_nonseparable_tensor_coupling|f17_cubic_chain_period17_quadratic_extension_resident_carrier)([[:space:]]|$)' "$oracle_path"; then
  echo "independent M120 oracle imports production or its phase backend" >&2
  exit 1
fi

"$python" - "$source_path" "$oracle_path" <<'PY'
import ast
import sys
from pathlib import Path


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"missing function {name}")


production_path, oracle_path = sys.argv[1:]
production_tree = ast.parse(Path(production_path).read_text(encoding="utf-8"), production_path)
oracle_tree = ast.parse(Path(oracle_path).read_text(encoding="utf-8"), oracle_path)
projection = function_node(production_tree, "project_boundary")
calls = [
    child.func.attr if isinstance(child.func, ast.Attribute) else (
        child.func.id if isinstance(child.func, ast.Name) else ""
    )
    for child in ast.walk(projection)
    if isinstance(child, ast.Call)
]
if calls.count("split_to_full") != 1:
    raise SystemExit("M120 projection must contain exactly one full scalar lift")
if "validate_program" not in calls:
    raise SystemExit("M120 projection does not validate the public runtime program")

accepted = {
    "execute_transaction",
    "project_boundary",
    "resident_butterfly_contract",
    "apply_resident_interface_butterflies",
}
for name in accepted:
    node = function_node(production_tree, name)
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        called = child.func.attr if isinstance(child.func, ast.Attribute) else (
            child.func.id if isinstance(child.func, ast.Name) else ""
        )
        if called in {"transfer_contract", "compact_transfer_boundary"}:
            raise SystemExit(f"accepted M120 function {name} calls the old dense row transfer")

butterfly = function_node(production_tree, "apply_resident_interface_butterflies")
names = {child.id for child in ast.walk(butterfly) if isinstance(child, ast.Name)}
if "target_assignment" in names or "source_assignment" in names:
    raise SystemExit("accepted M120 butterfly contains source-target enumeration")

required_oracle = {
    "gray_histogram",
    "butterfly_boundary",
    "independent_factor_restoration",
    "interface_rank_certificate",
    "rank_mod103",
}
present = {node.name for node in ast.walk(oracle_tree) if isinstance(node, ast.FunctionDef)}
missing = required_oracle - present
if missing:
    raise SystemExit(f"independent M120 oracle lacks {sorted(missing)}")
PY

"$python" -m py_compile "$source_path" "$oracle_path" "$benchmark_path"
nice -n 10 "$python" -X dev "$source_path" >"$result" 2>"$evidence_dir/m120.qualifier.result.stderr"
test ! -s "$evidence_dir/m120.qualifier.result.stderr"
nice -n 10 "$python" -X dev "$oracle_path" >"$oracle_result" 2>"$evidence_dir/m120.qualifier.oracle.stderr"
test ! -s "$evidence_dir/m120.qualifier.oracle.stderr"
nice -n 10 "$python" -X dev "$benchmark_path" >"$benchmark_result" 2>"$evidence_dir/m120.qualifier.benchmark.stderr"
test ! -s "$evidence_dir/m120.qualifier.benchmark.stderr"

cmp "$result" "$expected_path"
cmp "$oracle_result" "$oracle_expected_path"
test "$(sha256sum "$result" | cut -d' ' -f1)" = "$(jq -r '.reference_full_result_sha256' "$provenance_path")"
test "$(sha256sum "$oracle_result" | cut -d' ' -f1)" = "$(jq -r '.reference_full_oracle_sha256' "$provenance_path")"

jq -e '
  .result == "PASS_EXACT_TRANSITION_WORK_REPAIR_WITH_UNCHANGED_FULL_SEPARATOR_RANK"
  and .classification_candidate == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level_candidate == "PACKAGE_SELF_REVIEW"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .restoration_scope == "ACTUAL_BORROWED_FACTOR_CARRIER_FORWARD_PHASE_ACTIONS_REVERSED_AND_SEED_UNLOADED_TO_ORIGINAL_ZERO_BACKING"
  and .transient_butterfly_projection_buffer_restoration_class == "NO_RESTORATION_CLAIM"
  and .execution_scope == "LINUX_DIRECT_PROCESS_SOFTWARE"
  and .historical_m119_evidence_modified == false
  and .accepted_path_uses_actual_resident_factor_cells
  and (.accepted_path_dense_transfer_matrix_materialized|not)
  and (.accepted_path_source_target_pair_enumeration|not)
  and .accepted_path_butterfly_root_actions == [4,24,96]
  and .accepted_path_butterfly_additions == [8,48,192]
  and .accepted_path_row_separator_message_widths == [4,8,16]
  and .accepted_path_final_full_lifts_per_transaction == 1
  and (.intermediate_factor_or_frontier_values_projected|not)
  and .retained_inverse_history_bytes == 0
  and (.cases|length) == 6
  and all(.cases[];
    .boundary_agreement
    and .restored_exactly
    and .same_backing
    and .butterfly_root_count_exact
    and .butterfly_addition_count_exact
    and (.resources.four_to_the_n_source_target_transitions_materialized|not)
    and (.resources.dense_transfer_matrix_materialized|not)
    and (.resources.butterfly_layer_history_retained|not)
    and (.resources.transient_butterfly_projection_buffers_retained_after_projection|not)
    and (.resources.transient_butterfly_projection_buffers_restored_by_inverse|not)
    and .resources.transient_butterfly_projection_buffer_restoration_class == "NO_RESTORATION_CLAIM"
  )
  and [.cases[].separator_certificate.exact_rank_over_q_zeta17] == [4,4,8,8,16,16]
  and all(.restoration_reuse[];
    .same_original_backing
    and .fresh_restored_reuse_equal
    and (.baseline_reload|not)
    and .retained_inverse_history_bytes == 0
  )
  and .controls.premature_projection_rejected
  and .controls.omitted_butterfly_stage_changes_boundary
  and .controls.resident_vertical_factor_mutation_changes_boundary
  and .controls.mutated_factor_reverted_before_inverse_and_carrier_restored
  and .controls.wrong_projection_family_rejected
  and .controls.public_plan_excludes_runtime_weights_and_boundary
  and .controls.wrong_plan_fingerprint_rejected
  and .controls.wrong_projection_fingerprint_rejected
  and .controls.one_zero_weight_separator_rank_halves
  and .controls.one_zero_weight_separator_changes_boundary
  and .controls.false_separator_rank_cap_rejected
  and .controls.projection_guard_restored
  and .controls.null_carrier_rejected
  and .controls.missing_inverse_leaves_resident_state
  and .controls.wrong_inverse_exponent_fails_restoration
  and .controls.reordered_noncommuting_inverse_fails
  and .controls.resident_mutation_detected
  and .controls.snapshot_reload_absent
  and .matched_classical.strongest_evaluated_row_recurrence == "IDENTICAL_EXACT_KRONECKER_BUTTERFLY_ON_2_TO_N_Q_ZETA17_MESSAGES"
  and (.matched_classical.exact_best_order_treewidth_or_matchgate_reduction_proven_exhausted|not)
  and (.matched_classical.comparison_establishes_advantage|not)
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.not_established|index("REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and (.imports_production_m120_or_m119|not)
  and (.imports_phase_backend|not)
  and .all_boundaries_reconstructed_from_gray_histograms
  and .all_factor_cells_restore_exactly_on_same_backing
  and .all_interface_ranks_reexecuted_from_explicit_f103_matrices
  and .all_zero_weight_rank_halving_controls_pass
  and .all_operation_counts_reexecuted
  and (.cases|length) == 6
  and all(.cases[];
    .butterfly_gray_boundary_agreement
    and .exact_operation_counts
    and .column_stage_reorder_agrees
    and .omitted_stage_changes_boundary
    and .vertical_weight_mutation_changes_boundary
    and .factor_restoration.forward_changed_seed
    and .factor_restoration.seed_restored_exactly
    and .factor_restoration.unload_restored_zero_backing
    and .factor_restoration.same_backing
    and .rank_certificate.kronecker_determinant_nonzero
  )
  and [.cases[].rank_certificate.explicit_matrix_rank_mod103] == [4,4,8,8,16,16]
  and [.cases[].rank_certificate.one_zero_weight_matrix_rank_mod103] == [2,2,4,4,8,8]
  and (.broader_matchgate_holographic_add_mps_or_boundary_specific_reduction_ruled_out|not)
' "$oracle_result" >/dev/null

jq -e '
  .result == "PASS"
  and .warm_execution
  and (.cases|length) == 3
  and all(.cases[];
    .all_boundaries_equal
    and .restoring_carrier_restored_after_timing
    and .resident_phase_factor_butterfly_boundary_only.median_ns > 0
    and .compact_descriptor_butterfly_boundary_only.median_ns > 0
    and .gray_delta_global_histogram_boundary_only.median_ns > 0
    and .full_restoring_phase_transaction.median_ns > 0
  )
  and .resident_and_descriptor_interface_recurrences_are_identical
  and .resident_and_descriptor_row_diagonal_generation_are_not_operation_matched
  and .timing_is_observational_not_used_for_advantage_claim
  and .rss_is_process_wide_not_path_attributed
  and (.catvm_boundary_used|not)
' "$benchmark_result" >/dev/null

"$python" - "$result" "$oracle_result" "$m119_expected_path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    production = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    oracle = json.load(handle)
with open(sys.argv[3], encoding="utf-8") as handle:
    historical = json.load(handle)
for actual, reference, prior in zip(
    production["cases"],
    oracle["cases"],
    historical["cases"],
    strict=True,
):
    for key in ("n", "family", "plan_fingerprint", "unary_weights", "edge_weights"):
        if actual[key] != reference[key]:
            raise SystemExit(f"M120 production/oracle {key} mismatch")
    if actual["boundary"] != reference["canonical_boundary"]:
        raise SystemExit("M120 production/oracle boundary mismatch")
    if actual["boundary"] != prior["boundary"]:
        raise SystemExit("M120 butterfly changed the frozen M119 boundary")
    if actual["butterfly_stats"]["butterfly_root_actions"] != reference["butterfly_stats"]["butterfly_root_actions"]:
        raise SystemExit("M120 production/oracle root action count mismatch")
    if actual["butterfly_stats"]["butterfly_additions"] != reference["butterfly_stats"]["butterfly_additions"]:
        raise SystemExit("M120 production/oracle butterfly addition count mismatch")
PY

echo "QUALIFIED_F17_RUNTIME_WEIGHTED_GRID_KRONECKER_BUTTERFLY_CLOSURE"
