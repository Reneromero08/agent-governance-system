#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_runtime_weighted_grid_phase_factor_closure.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_runtime_weighted_grid_phase_factor_closure.py"
oracle_path="$frontier_dir/f17_runtime_weighted_grid_phase_factor_closure_oracle.py"
benchmark_source="$frontier_dir/benchmark_f17_runtime_weighted_grid_phase_factor_closure.py"
m118_path="$frontier_dir/f17_variable_rank_nonseparable_tensor_coupling.py"
m116_path="$frontier_dir/f17_cubic_chain_period17_quadratic_extension_resident_carrier.py"
m118_provenance="$frontier_dir/F17_VARIABLE_RANK_NONSEPARABLE_TENSOR_COUPLING_PROVENANCE.json"
expected_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_PHASE_FACTOR_CLOSURE_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_PHASE_FACTOR_CLOSURE_ORACLE_RESULTS.json"
benchmark_expected_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_PHASE_FACTOR_CLOSURE_BENCHMARK.json"
provenance_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_PHASE_FACTOR_CLOSURE_PROVENANCE.json"
review_path="$frontier_dir/F17_RUNTIME_WEIGHTED_GRID_PHASE_FACTOR_CLOSURE_INDEPENDENT_REVIEW.md"
qualifier_path="$frontier_dir/qualify_f17_runtime_weighted_grid_phase_factor_closure.sh"
result="$evidence_dir/m119.qualifier.result.full.json"
oracle_result="$evidence_dir/m119.qualifier.oracle.full.json"
benchmark_result="$evidence_dir/m119.qualifier.benchmark.observed.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in cmp git jq nice rg sha256sum; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path" "$benchmark_expected_path" "$provenance_path"

scientific_parent=$(jq -r '.scientific_source_parent' "$provenance_path")
test "$scientific_parent" = "7bdbac7a8eb047b82bcd9171402188ae665a2cef"
git -C "$repo" cat-file -e "$scientific_parent^{commit}"
git -C "$repo" merge-base --is-ancestor "$scientific_parent" HEAD

for sealed_path in \
  "$source_path" \
  "$oracle_path" \
  "$benchmark_source" \
  "$m118_path" \
  "$m116_path" \
  "$m118_provenance" \
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

if rg -n '^[[:space:]]*(from|import)[[:space:]]+(f17_runtime_weighted_grid_phase_factor_closure|f17_variable_rank_nonseparable_tensor_coupling|f17_cubic_chain_period17_quadratic_extension_resident_carrier)([[:space:]]|$)' "$oracle_path"; then
  echo "independent M119 oracle imports production or its phase backend" >&2
  exit 1
fi

"$python" - "$source_path" "$oracle_path" <<'PY'
import ast
import sys
from pathlib import Path


def function_node(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"missing function {name} in {path}")


production, oracle = sys.argv[1:]
projection = function_node(production, "project_boundary")
projection_calls = [
    child.func.attr if isinstance(child.func, ast.Attribute) else (
        child.func.id if isinstance(child.func, ast.Name) else ""
    )
    for child in ast.walk(projection)
    if isinstance(child, ast.Call)
]
if projection_calls.count("split_to_full") != 1:
    raise SystemExit("production final projection must contain exactly one full scalar lift")
if "validate_program" not in projection_calls:
    raise SystemExit("production projection does not validate the runtime program")

required_oracle = {
    "streamed_dense_histogram",
    "gray_delta_histogram",
    "character_transfer_histogram",
    "independent_factor_restoration",
    "separator_certificate",
    "reduced_energy_mtbdd",
}
oracle_tree = ast.parse(Path(oracle).read_text(encoding="utf-8"), filename=oracle)
present = {node.name for node in ast.walk(oracle_tree) if isinstance(node, ast.FunctionDef)}
missing = required_oracle - present
if missing:
    raise SystemExit(f"independent oracle is missing functions: {sorted(missing)}")
PY

"$python" -m py_compile "$source_path" "$oracle_path" "$benchmark_source"
nice -n 10 "$python" -X dev "$source_path" >"$result" 2>"$evidence_dir/m119.qualifier.result.stderr"
test ! -s "$evidence_dir/m119.qualifier.result.stderr"
nice -n 10 "$python" -X dev "$oracle_path" >"$oracle_result" 2>"$evidence_dir/m119.qualifier.oracle.stderr"
test ! -s "$evidence_dir/m119.qualifier.oracle.stderr"
nice -n 10 "$python" -X dev "$benchmark_source" >"$benchmark_result" 2>"$evidence_dir/m119.qualifier.benchmark.stderr"
test ! -s "$evidence_dir/m119.qualifier.benchmark.stderr"

cmp "$result" "$expected_path"
cmp "$oracle_result" "$oracle_expected_path"
test "$(sha256sum "$result" | cut -d' ' -f1)" = "$(jq -r '.reference_full_result_sha256' "$provenance_path")"
test "$(sha256sum "$oracle_result" | cut -d' ' -f1)" = "$(jq -r '.reference_full_oracle_sha256' "$provenance_path")"

jq -e '
  .result == "PASS_GROWING_TREEWIDTH_NEGATIVE_RESOURCE_DIAGNOSTIC"
  and .classification_candidate == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level_candidate == "PACKAGE_SELF_REVIEW"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .execution_scope == "LINUX_DIRECT_PROCESS_SOFTWARE"
  and (.accepted_path_dense_assignment_tensor_materialized|not)
  and (.accepted_path_global_assignment_enumeration|not)
  and .accepted_path_row_separator_message_widths == [4,8,16]
  and .accepted_path_row_assignments_evaluated_per_row == [4,8,16]
  and .accepted_path_source_target_transitions_per_row_interface == [16,64,256]
  and .retained_inverse_history_bytes == 0
  and (.intermediate_factor_or_transfer_values_projected|not)
  and (.cases|length) == 6
  and all(.cases[];
    .boundary_agreement
    and .restored_exactly
    and .same_backing
    and .runtime_weights_bound_after_topology_compile
    and .separator_certificate.certifies_actual_row_transfer_interface
    and .separator_certificate.determinant_mod103_nonzero
    and (.direct_zero_field_planar_ising_pfaffian_applicable|not)
    and .resources.stored_phase_family_and_fingerprint_strings_counted_as_utf8_payload_bits
    and .resources.compile_and_bind.runtime_weight_formula_evaluations > 0
  )
  and [.cases[].separator_certificate.exact_rank_over_q_zeta17] == [4,4,8,8,16,16]
  and [.cases[].grid_treewidth] == [2,2,3,3,4,4]
  and all(.restoration_reuse[];
    .fresh_restored_reuse_boundary_equal
    and .fresh_restored_reuse_full_nonmetadata_stats_equal
    and .fresh_restored_reuse_separator_certificate_equal
    and .same_original_backing
    and (.baseline_reload|not)
    and .retained_inverse_history_bytes == 0
  )
  and (.controls.compiled_plan_contains_runtime_weights|not)
  and .controls.premature_projection_rejected
  and .controls.missing_inverse_leaves_resident_state
  and .controls.wrong_inverse_exponent_fails_restoration
  and .controls.reordered_noncommuting_unary_prepare_inverse_fails
  and .controls.resident_mutation_detected
  and .controls.null_carrier_rejected
  and .controls.wrong_plan_fingerprint_rejected
  and .controls.wrong_projection_family_rejected
  and .controls.wrong_projection_fingerprint_rejected
  and .controls.projection_guard_carrier_restored
  and .controls.runtime_weight_mutation_changes_boundary
  and .controls.one_separator_edge_removed_changes_boundary
  and .controls.forced_rank_below_certificate_rejected
  and .controls.snapshot_reload_absent
  and (.matched_classical.evaluated_compact_baseline_set_not_proven_exhaustive_or_pareto_optimal|length) == 3
  and .matched_classical.pareto_baseline_set == null
  and (.matched_classical.three_order_mtbdd_sweep_claims_order_optimality|not)
  and (.matched_classical.comparison_establishes_advantage|not)
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.not_established|index("REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and (.imports_production_m119|not)
  and (.imports_m116_or_m118_backend|not)
  and .all_histograms_agree
  and .all_boundaries_reconstructed_independently
  and .all_inverse_phase_streams_restore
  and .all_factor_cells_restore_exactly_on_same_backing
  and .all_separator_ranks_certified
  and .all_direct_zero_field_pfaffian_checks_inapplicable
  and (.broader_matchgate_or_holographic_reduction_ruled_out|not)
  and (.evaluated_baseline_set_proven_exhaustive_or_pareto_optimal|not)
  and (.cases|length) == 6
  and all(.cases[];
    .histograms_agree
    and .histogram_total_assignments == .expected_total_assignments
    and .independent_factor_restoration.seed_restored_exactly
    and .independent_factor_restoration.unload_restored_zero_backing
    and .independent_factor_restoration.same_backing
    and .separator_certificate.certifies_actual_row_transfer_interface
    and all(.three_order_observed_reduced_energy_mtbdd_sweep[];
      .full_assignment_tree_built_by_this_oracle
      and (.order_optimality_claimed|not)
      and .full_assignment_leaves_visited == .full_binary_tree_nonterminal_nodes + 1
    )
  )
  and [.cases[].gray_delta_streaming_stats.assignments_streamed] == [16,16,512,512,65536,65536]
  and [.cases[].separator_certificate.exact_rank_over_q_zeta17] == [4,4,8,8,16,16]
' "$oracle_result" >/dev/null

jq -e '
  .result == "PASS"
  and .warm_execution
  and (.cases|length) == 3
  and all(.cases[];
    .all_boundaries_equal
    and .phase_carrier_restored_after_timing
    and .phase_restoring_transaction.median_ns > 0
    and .compact_transfer_boundary_only.median_ns > 0
    and .gray_delta_global_histogram_boundary_only.median_ns > 0
  )
  and .timed_paths_are_not_operation_matched
  and .timing_is_observational_not_used_for_advantage_claim
  and .rss_is_process_wide_not_path_attributed
  and (.catvm_boundary_used|not)
' "$benchmark_result" >/dev/null

"$python" - "$result" "$oracle_result" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    production = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    oracle = json.load(handle)

for production_case, oracle_case in zip(production["cases"], oracle["cases"], strict=True):
    for key in ("n", "family", "plan_fingerprint", "unary_weights", "edge_weights"):
        if production_case[key] != oracle_case[key]:
            raise SystemExit(f"production/oracle {key} mismatch")
    if production_case["boundary"] != oracle_case["canonical_boundary"]:
        raise SystemExit("production/oracle boundary mismatch")
    if production_case["separator_certificate"]["exact_rank_over_q_zeta17"] != oracle_case["separator_certificate"]["exact_rank_over_q_zeta17"]:
        raise SystemExit("production/oracle separator rank mismatch")
PY

echo "QUALIFIED_F17_RUNTIME_WEIGHTED_GRID_PHASE_FACTOR_CLOSURE"
