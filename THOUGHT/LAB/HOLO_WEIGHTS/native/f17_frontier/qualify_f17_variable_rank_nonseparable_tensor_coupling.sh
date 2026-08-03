#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_variable_rank_nonseparable_tensor_coupling.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_variable_rank_nonseparable_tensor_coupling.py"
oracle_path="$frontier_dir/f17_variable_rank_nonseparable_tensor_coupling_oracle.py"
benchmark_source="$frontier_dir/benchmark_f17_variable_rank_nonseparable_tensor_coupling.py"
predecessor_path="$frontier_dir/f17_three_shear_relative_hermitian_trace_feedback.py"
expected_path="$frontier_dir/F17_VARIABLE_RANK_NONSEPARABLE_TENSOR_COUPLING_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_VARIABLE_RANK_NONSEPARABLE_TENSOR_COUPLING_ORACLE_RESULTS.json"
benchmark_expected_path="$frontier_dir/F17_VARIABLE_RANK_NONSEPARABLE_TENSOR_COUPLING_BENCHMARK.json"
provenance_path="$frontier_dir/F17_VARIABLE_RANK_NONSEPARABLE_TENSOR_COUPLING_PROVENANCE.json"
review_path="$frontier_dir/F17_VARIABLE_RANK_NONSEPARABLE_TENSOR_COUPLING_INDEPENDENT_REVIEW.md"
qualifier_path="$frontier_dir/qualify_f17_variable_rank_nonseparable_tensor_coupling.sh"
result="$evidence_dir/m118.qualifier.result.full.json"
oracle_result="$evidence_dir/m118.qualifier.oracle.full.json"
benchmark_result="$evidence_dir/m118.qualifier.benchmark.observed.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in cmp git jq nice rg sha256sum; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path" "$benchmark_expected_path" "$provenance_path"

scientific_parent=$(jq -r '.scientific_source_parent' "$provenance_path")
test "$scientific_parent" = "fc8825af6eb392e5a4d143bcebd74b5354f4841d"
git -C "$repo" cat-file -e "$scientific_parent^{commit}"
git -C "$repo" merge-base --is-ancestor "$scientific_parent" HEAD

for sealed_path in \
  "$source_path" \
  "$oracle_path" \
  "$benchmark_source" \
  "$predecessor_path" \
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

if rg -n '^[[:space:]]*(from|import)[[:space:]]+(f17_variable_rank_nonseparable_tensor_coupling|f17_cubic_chain_period17_quadratic_extension_resident_carrier)([[:space:]]|$)' "$oracle_path"; then
  echo "independent oracle imports production M118 or the M116 pair backend" >&2
  exit 1
fi

"$python" - "$source_path" "$oracle_path" <<'PY'
import ast
import sys
from pathlib import Path


def function_calls(path: str, name: str) -> list[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            calls = []
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                target = child.func
                calls.append(
                    target.attr if isinstance(target, ast.Attribute) else (
                        target.id if isinstance(target, ast.Name) else ""
                    )
                )
            return calls
    raise SystemExit(f"missing function {name} in {path}")


production, oracle = sys.argv[1:]
for name in ("apply_phase_gate", "apply_shear", "apply_gate"):
    if "split_to_full" in function_calls(production, name):
        raise SystemExit(f"accepted internal operation lifts a full cyclotomic value: {name}")
if function_calls(production, "project_boundary").count("split_to_full") != 1:
    raise SystemExit("production final projection must contain exactly one full scalar lift")
required_oracle = {
    "multiply", "phase_gate", "shear", "determinant", "exact_rank",
    "finite_rank", "compact_boundary", "three_product_boundary",
    "outer_assignment_boundary", "reverse",
}
tree = ast.parse(Path(oracle).read_text(encoding="utf-8"), filename=oracle)
present = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
missing = required_oracle - present
if missing:
    raise SystemExit(f"independent oracle is missing functions: {sorted(missing)}")
PY

"$python" -m py_compile "$source_path" "$oracle_path" "$benchmark_source"
nice -n 10 "$python" -X dev "$source_path" >"$result" 2>"$evidence_dir/m118.qualifier.result.stderr"
test ! -s "$evidence_dir/m118.qualifier.result.stderr"
nice -n 10 "$python" -X dev "$oracle_path" >"$oracle_result" 2>"$evidence_dir/m118.qualifier.oracle.stderr"
test ! -s "$evidence_dir/m118.qualifier.oracle.stderr"
nice -n 10 "$python" -X dev "$benchmark_source" --output "$benchmark_result" 2>"$evidence_dir/m118.qualifier.benchmark.stderr"
test ! -s "$evidence_dir/m118.qualifier.benchmark.stderr"

cmp "$result" "$expected_path"
cmp "$oracle_result" "$oracle_expected_path"
test "$(sha256sum "$result" | cut -d' ' -f1)" = "$(jq -r '.reference_full_result_sha256' "$provenance_path")"
test "$(sha256sum "$oracle_result" | cut -d' ' -f1)" = "$(jq -r '.reference_full_oracle_sha256' "$provenance_path")"

jq -e '
  .result == "PASS_NEGATIVE_RESOURCE_DIAGNOSTIC"
  and .classification_candidate == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level_candidate == "PACKAGE_SELF_REVIEW"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .logical_tensor_shape == [2,2,2,2]
  and .logical_phase_cells == 16
  and .logical_integer_coordinates == 256
  and (.intermediate_amplitudes_projected_on_accepted_path|not)
  and .accepted_path_split_to_full_lifts == 1
  and .retained_inverse_history_bytes == 0
  and [.cases[].boundary] == [
    [9,2,1,4,2,1,0,1,2,0,0,2,0,0,0,0],
    [9,0,2,0,3,0,4,0,2,1,0,0,0,1,0,2]
  ]
  and all(.cases[];
    .boundary_agreement
    and .final_all_two_by_two_cuts_rank_four
    and .final_all_one_site_cuts_rank_two
    and .restored_exactly
    and .same_backing
    and [.rank_trace[].natural_tt_ranks] == [
      [1,1,1],[2,2,1],[2,2,1],[2,2,1],[2,4,2],[2,4,2]
    ]
    and .compact_classical_stats.pair_multiplications == 0
    and .compact_classical_stats.uncoalesced_nonconstant_character_terms == 9
    and .dense_phase_named_component_maxima_sum_bits > .compact_classical_named_component_maxima_sum_bits
  )
  and [.cases[].compact_classical_stats.coalesced_nonconstant_character_terms] == [8,7]
  and all(.controls[]; .)
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and .restoration_reuse_case.fresh_restored_reuse_rank_trace_and_full_nonmetadata_arithmetic_signature_equal
  and (.restoration_reuse_case.baseline_reload|not)
  and .matched_classical.optimal_factor_graph_treewidth == 2
  and .matched_classical.maximum_live_phase_cells == 4
  and .matched_classical.equal_or_lower_resource_signature_available
  and (.matched_classical.comparison_establishes_advantage|not)
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and (.imports_production_m118|not)
  and (.imports_m116_pair_backend|not)
  and .representation == "CANONICAL_16_INTEGER_POWER_BASIS"
  and .all_boundaries_agree
  and .all_closed_form_factor_contractions_match_outer_assignment_sums
  and .all_final_cut_ranks_four_exactly
  and .all_mod103_cut_ranks_four
  and .all_restore_and_unload_exactly
  and .all_same_backing
  and all(.controls[]; .)
  and [.cases[].final_determinant_mod_103] == [
    {"AB_CD":59,"AC_BD":42,"AD_BC":86},
    {"AB_CD":90,"AC_BD":65,"AD_BC":78}
  ]
' "$oracle_result" >/dev/null

jq -e '
  .result == "PASS"
  and .boundaries_equal
  and .warmups_per_path == 20
  and .phase_path.samples == 101
  and .compact_factor_path.samples == 101
  and .phase_path.minimum_ns > 0
  and .compact_factor_path.minimum_ns > 0
  and .phase_path_scope == "FORWARD_FINAL_BOUNDARY_INVERSE_RESTORATION"
  and .compact_factor_path_scope == "FINAL_BOUNDARY_EVALUATION_ONLY"
  and .timed_paths_are_not_operation_matched
  and .rss_is_process_wide_not_path_attributed
  and .timing_is_local_observational_not_used_for_advantage_claim
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
    if production_case["family"] != oracle_case["family"]:
        raise SystemExit("family order mismatch")
    if production_case["boundary"] != oracle_case["boundary"]:
        raise SystemExit(f"boundary mismatch for {production_case['family']}")
    if production_case["rank_trace"] != oracle_case["rank_trace"]:
        raise SystemExit(f"rank trace mismatch for {production_case['family']}")
PY

echo "QUALIFIED_F17_VARIABLE_RANK_NONSEPARABLE_TENSOR_COUPLING"
