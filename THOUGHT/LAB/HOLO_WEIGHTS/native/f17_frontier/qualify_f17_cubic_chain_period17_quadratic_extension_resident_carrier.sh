#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_quadratic_extension_resident_carrier.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_quadratic_extension_resident_carrier.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_quadratic_extension_resident_carrier_oracle.py"
predecessor_path="$frontier_dir/f17_cubic_chain_period17_direct_real_hermitian.py"
predecessor_oracle_path="$frontier_dir/f17_cubic_chain_period17_direct_real_hermitian_oracle.py"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_QUADRATIC_EXTENSION_RESIDENT_CARRIER_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_QUADRATIC_EXTENSION_RESIDENT_CARRIER_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_QUADRATIC_EXTENSION_RESIDENT_CARRIER_PROVENANCE.json"
review_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_QUADRATIC_EXTENSION_RESIDENT_CARRIER_INDEPENDENT_REVIEW.md"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_quadratic_extension_resident_carrier.sh"
result="$evidence_dir/m116.qualifier.result.full.json"
oracle_result="$evidence_dir/m116.qualifier.oracle.full.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in cmp git jq nice rg sha256sum; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path" "$provenance_path"

scientific_parent=$(jq -r '.scientific_source_parent' "$provenance_path")
test "$scientific_parent" = "86fd1c944a139996709b04714be6713fc64726cd"
git -C "$repo" cat-file -e "$scientific_parent^{commit}"
git -C "$repo" merge-base --is-ancestor "$scientific_parent" HEAD

for sealed_path in \
  "$source_path" \
  "$oracle_path" \
  "$predecessor_path" \
  "$predecessor_oracle_path" \
  "$qualifier_path" \
  "$expected_path" \
  "$oracle_expected_path" \
  "$review_path"
do
  sealed_name=$(basename "$sealed_path")
  sealed_expected=$(jq -r --arg name "$sealed_name" \
    '.files[$name] // empty' "$provenance_path")
  test -n "$sealed_expected"
  test "$(sha256sum "$sealed_path" | cut -d' ' -f1)" = "$sealed_expected"
done

if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_period17_quadratic_extension_resident_carrier([[:space:]]|$)' \
  "$oracle_path"; then
  echo "independent oracle imports production M116 module" >&2
  exit 1
fi

"$python" - "$source_path" "$oracle_path" <<'PY'
import ast
import sys
from pathlib import Path


def function_calls(path: str, function: str) -> list[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            calls = []
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                target = child.func
                name = target.attr if isinstance(target, ast.Attribute) else (
                    target.id if isinstance(target, ast.Name) else ""
                )
                calls.append(name)
            return calls
    raise SystemExit(f"missing function {function} in {path}")


production = sys.argv[1]
oracle = sys.argv[2]
project_calls = function_calls(production, "split_project_boundary")
if project_calls.count("split_to_full") != 1:
    raise SystemExit("production projection must have exactly one scalar full lift")
for function in ("populate_forward", "restore_forward", "split_multiply"):
    calls = function_calls(production, function)
    if "split_to_full" in calls or "ring_multiply" in calls:
        raise SystemExit(f"forbidden full lift or multiply in {function}: {calls!r}")
for function in ("pair_multiply", "convert_vector", "project"):
    calls = function_calls(oracle, function)
    if "ring_multiply" in calls:
        raise SystemExit(f"oracle pair schedule called full ring multiply in {function}")
PY

"$python" -m py_compile "$source_path" "$oracle_path"
nice -n 10 "$python" -X dev "$source_path" \
  >"$result" 2>"$evidence_dir/m116.qualifier.result.stderr"
test ! -s "$evidence_dir/m116.qualifier.result.stderr"
nice -n 10 "$python" -X dev "$oracle_path" "$result" \
  >"$oracle_result" 2>"$evidence_dir/m116.qualifier.oracle.stderr"
test ! -s "$evidence_dir/m116.qualifier.oracle.stderr"

cmp "$result" "$expected_path"
cmp "$oracle_result" "$oracle_expected_path"
test "$(sha256sum "$result" | cut -d' ' -f1)" = \
  "$(jq -r '.reference_full_result_sha256' "$provenance_path")"
test "$(sha256sum "$oracle_result" | cut -d' ' -f1)" = \
  "$(jq -r '.reference_full_oracle_sha256' "$provenance_path")"

jq -e '
  .result == "PASS"
  and .classification_candidate == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level_candidate == "PACKAGE_SELF_REVIEW"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .tested_periods == [1,64]
  and .representation_isomorphism
  and (.dimension_reducing_quotient|not)
  and .resident_integer_coordinate_count_per_element == 16
  and .forward_horner_construction_remains_full_cyclotomic
  and .inverse_rematerialization_remains_full_cyclotomic
  and (.resident_full_cyclotomic_vector_retained_after_conversion|not)
  and (.accepted_projection_full_vector_materialized|not)
  and .boundary_full_scalar_lifts_per_transaction == 1
  and .all_boundaries_equal_raw_horner
  and .all_cases_restore_exactly
  and .all_cases_use_one_boundary_full_lift
  and [.cases[].phase_named_component_maxima_sum_bits]
    == [94616,102400,3325225,3696278]
  and [.cases[].raw_horner_named_checkpoint_payload_bits]
    == [10005,10097,2790766,2901994]
  and [.cases[].split_resident_minus_comparable_full_resident_payload_bits]
    == [99,198,57,116]
  and all(.cases[];
    .raw_horner_boundary_equal
    and .restored_exactly
    and .same_backing
    and .phase_minus_raw_horner_named_payload_bits > 0
    and .phase_stats.split_to_full_boundary_lifts == 1
    and .inverse_rematerialization_stats.split_to_full_boundary_lifts == 0
  )
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and .restoration_reuse_case.fresh_restored_reuse_phase_signature_equal
  and (.restoration_reuse_case.baseline_reload|not)
  and all(.carrier_controls[]; .)
  and all(.algebra_controls[]; .)
  and .matched_classical.identical_two_by_eight_quadratic_extension_carrier_available
  and (.matched_classical.comparison_establishes_advantage|not)
  and (.not_established|index("DIMENSION_OR_RANK_REDUCTION")) != null
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "SEPARATE_REFERENCE_PARITY"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and (.oracle_imports_production_m116_module|not)
  and all(.family_checks[]; all(.[]; .))
  and all(.case_checks[];
    .boundary_sha256_equal
    and .raw_boundary_sha256_equal
    and .phase_resource_tuple_equal
    and .inverse_resource_tuple_equal
    and .raw_resource_tuple_equal
    and .checkpoint_equal
    and .search_temporary_equal
    and .retained_tables_equal
    and .named_total_equal
    and .comparable_resident_delta_equal
    and .inverse_output_equal
    and .pair_restoration_exact
    and .semantic_power_pair_matches_s_pair
  )
  and all(.restoration_checks[]; .)
  and all(.algebra_checks[]; .)
  and all(.scope_checks[]; .)
  and all(.public_table_checks[]; .)
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "QUALIFIED_F17_CUBIC_CHAIN_PERIOD17_QUADRATIC_EXTENSION_RESIDENT_CARRIER"
