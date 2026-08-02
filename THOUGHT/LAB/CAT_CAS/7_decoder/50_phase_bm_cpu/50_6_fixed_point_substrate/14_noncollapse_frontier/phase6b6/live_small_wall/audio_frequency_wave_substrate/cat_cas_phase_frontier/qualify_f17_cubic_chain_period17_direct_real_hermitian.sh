#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_cubic_chain_period17_direct_real_hermitian.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_cubic_chain_period17_direct_real_hermitian.py"
oracle_path="$frontier_dir/f17_cubic_chain_period17_direct_real_hermitian_oracle.py"
predecessor_path="$frontier_dir/f17_cubic_chain_period17_streamed_real_autocorrelation.py"
predecessor_oracle_path="$frontier_dir/f17_cubic_chain_period17_streamed_real_autocorrelation_oracle.py"
qualifier_path="$frontier_dir/qualify_f17_cubic_chain_period17_direct_real_hermitian.sh"
expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_DIRECT_REAL_HERMITIAN_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_DIRECT_REAL_HERMITIAN_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_DIRECT_REAL_HERMITIAN_PROVENANCE.json"
review_path="$frontier_dir/F17_CUBIC_CHAIN_PERIOD17_DIRECT_REAL_HERMITIAN_INDEPENDENT_REVIEW.md"
result="$evidence_dir/result.full.json"
oracle_result="$evidence_dir/oracle.full.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in cmp git jq nice rg sha256sum; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path" "$provenance_path"

scientific_parent=$(jq -r '.scientific_source_parent' "$provenance_path")
test "$scientific_parent" = "2ed6ae2a906a6873b58a77cabaf886377a8234bb"
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

if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_cubic_chain_period17_direct_real_hermitian([[:space:]]|$)' \
  "$oracle_path"; then
  echo "independent oracle imports production direct successor" >&2
  exit 1
fi

"$python" - "$source_path" "$oracle_path" <<'PY'
import ast
import sys
from pathlib import Path


def forbidden_calls(path: str, functions: tuple[str, ...], forbidden: set[str]) -> list[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    found = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in functions:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            target = child.func
            name = target.attr if isinstance(target, ast.Attribute) else (
                target.id if isinstance(target, ast.Name) else ""
            )
            if name in forbidden:
                found.append(f"{node.name}:{name}:{child.lineno}")
    return found


production_bad = forbidden_calls(
    sys.argv[1],
    ("direct_real_hermitian_term", "direct_real_vector_norm"),
    {"ring_multiply", "ring_conjugate", "full_to_real", "real_to_full"},
)
oracle_bad = forbidden_calls(
    sys.argv[2],
    ("quadratic_extension_norm", "direct_power_vector_norm"),
    {"ring_multiply", "conjugate", "full_to_power", "full_to_s"},
)
if production_bad or oracle_bad:
    raise SystemExit(
        "degree-16 operation entered accepted semantic path: "
        + repr((production_bad, oracle_bad))
    )
PY

"$python" -m py_compile "$source_path" "$oracle_path"
nice -n 10 "$python" -X dev "$source_path" \
  >"$result" 2>"$evidence_dir/result.stderr"
test ! -s "$evidence_dir/result.stderr"
nice -n 10 "$python" -X dev "$oracle_path" "$result" \
  >"$oracle_result" 2>"$evidence_dir/oracle.stderr"
test ! -s "$evidence_dir/oracle.stderr"

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
  and (.full_cyclotomic_aggregate_norm_constructed|not)
  and (.full_cyclotomic_per_element_norm_products_remain|not)
  and .direct_real_hermitian_generator_used
  and .all_direct_operation_counts_exact
  and all(.direct_formula_controls[]; .)
  and .direct_generator_public_plan_payload_bits == 0
  and .all_raw_horner_boundaries_equal
  and .all_prior_raw_recurrence_boundaries_equal
  and .all_cases_restore_exactly
  and (.all_phase_named_payloads_beat_raw_horner|not)
  and [.cases[].phase_named_component_maxima_sum_bits]
    == [93790,101475,3324441,3695435]
  and [.cases[].raw_horner_named_checkpoint_payload_bits]
    == [10005,10097,2790766,2901994]
  and [.cases[].phase_stats.streamed_real_norm_calls]
    == [5,5,93,98]
  and [.cases[].phase_stats.streamed_real_norm_terms]
    == [69,69,1101,1154]
  and [.cases[].phase_stats.direct_real_hermitian_calls]
    == [69,69,1101,1154]
  and [.cases[].phase_stats.direct_real_hermitian_coefficient_multiplications]
    == [9384,9384,149736,156944]
  and [.cases[].phase_stats.direct_real_hermitian_accumulation_additions]
    == [8763,8763,139827,146558]
  and [.cases[].phase_stats.direct_real_hermitian_coefficient_subtractions]
    == [552,552,8808,9232]
  and all(.cases[];
    .phase_stats.streamed_real_norm_full_cyclotomic_multiplications == 0
    and .phase_stats.direct_real_hermitian_full_cyclotomic_product_calls == 0
    and .phase_stats.direct_real_hermitian_materialized_conjugate_calls == 0
    and .phase_stats.maximum_direct_real_degree16_scratch_payload_bits == 0
    and .phase_stats.maximum_streamed_norm_full_product_payload_bits == 0
    and .phase_stats.full_to_real_conversions == 0
    and .restored_exactly
    and .same_backing
    and .raw_horner_boundary_equal
    and .prior_raw_recurrence_boundary_equal
    and .phase_minus_raw_horner_named_payload_bits > 0
  )
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and all(.carrier_controls[]; .)
  and all(.algebra_controls[]; .)
  and .matched_classical.identical_direct_real_hermitian_map_available
  and .matched_classical.same_136_integer_products_per_term
  and (.matched_classical.comparison_establishes_advantage|not)
  and .resource_law.direct_integer_products_counted
  and .resource_law.direct_generation_live_payload_counted
  and .resource_law.verification_full_products_excluded_and_reported
  and (.resource_law.whole_process_peak_bounded|not)
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.not_established|index("MACHINE_ENFORCED_NO_SMUGGLE_OR_CATVM_CUSTODY")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
  and .verification_level == "SEPARATE_REFERENCE_PARITY"
  and .restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and (.oracle_imports_production_direct_module|not)
  and (.oracle_semantic_path_uses_cyclic_autocorrelation|not)
  and (.oracle_semantic_path_constructs_degree16_product|not)
  and (.oracle_resource_schedule_constructs_degree16_product|not)
  and all(.family_checks[]; all(.[]; .))
  and all(.case_checks[];
    .boundary_sha256_equal
    and .raw_horner_boundary_sha256_equal
    and .phase_resource_tuple_equal
    and .inverse_resource_tuple_equal
    and .raw_horner_resource_tuple_equal
    and .phase_named_checkpoint_equal
    and .named_search_temporary_sum_equal
    and .named_component_total_equal
    and .inverse_output_exactly_equal
  )
  and all(.restoration_checks[]; .)
  and all(.mutation_checks[]; .)
  and all(.algebra_checks[]; .)
  and all(.compiled_table_checks[]; .)
  and all(.production_scope_checks[]; .)
  and all(.streamed_shape_checks[]; .)
  and all(.independent_controls[]; .)
  and all(.direct_case_checks[]; .)
  and .quadratic_extension_oracle_counts.schedule_parity_checks > 10000
  and .quadratic_extension_oracle_counts.semantic_calls
    == (.quadratic_extension_oracle_counts.semantic_power_multiplications / 4)
  and (.terminal|not)
' "$oracle_result" >/dev/null

echo "QUALIFIED_F17_CUBIC_CHAIN_PERIOD17_DIRECT_REAL_HERMITIAN"
