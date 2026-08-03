#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_three_shear_relative_hermitian_trace_feedback.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_three_shear_relative_hermitian_trace_feedback.py"
oracle_path="$frontier_dir/f17_three_shear_relative_hermitian_trace_feedback_oracle.py"
predecessor_path="$frontier_dir/f17_cubic_chain_period17_quadratic_extension_resident_carrier.py"
predecessor_oracle_path="$frontier_dir/f17_cubic_chain_period17_quadratic_extension_resident_carrier_oracle.py"
expected_path="$frontier_dir/F17_THREE_SHEAR_RELATIVE_HERMITIAN_TRACE_FEEDBACK_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_THREE_SHEAR_RELATIVE_HERMITIAN_TRACE_FEEDBACK_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_THREE_SHEAR_RELATIVE_HERMITIAN_TRACE_FEEDBACK_PROVENANCE.json"
review_path="$frontier_dir/F17_THREE_SHEAR_RELATIVE_HERMITIAN_TRACE_FEEDBACK_INDEPENDENT_REVIEW.md"
qualifier_path="$frontier_dir/qualify_f17_three_shear_relative_hermitian_trace_feedback.sh"
result="$evidence_dir/m117.qualifier.result.full.json"
oracle_result="$evidence_dir/m117.qualifier.oracle.full.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in cmp git jq nice rg sha256sum; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path" "$provenance_path"

scientific_parent=$(jq -r '.scientific_source_parent' "$provenance_path")
test "$scientific_parent" = "367716267b83f2bcbcb6c3cd3d52f6209f70a582"
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

if rg -n '^[[:space:]]*(from|import)[[:space:]]+f17_three_shear_relative_hermitian_trace_feedback([[:space:]]|$)' \
  "$oracle_path"; then
  echo "independent oracle imports production M117 module" >&2
  exit 1
fi

"$python" - "$source_path" "$oracle_path" <<'PY'
import ast
import sys
from pathlib import Path


def calls(path: str, function: str) -> list[str]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            result = []
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                target = child.func
                result.append(
                    target.attr if isinstance(target, ast.Attribute) else (
                        target.id if isinstance(target, ast.Name) else ""
                    )
                )
            return result
    raise SystemExit(f"missing function {function} in {path}")


production, oracle = sys.argv[1:]
if calls(production, "relative_hermitian_trace").count("real_multiply") != 3:
    raise SystemExit("production relative trace is not the three-product schedule")
injection_calls = calls(production, "fixed_root_injection")
if injection_calls.count("real_multiply") != 2 or "split_multiply" in injection_calls:
    raise SystemExit("production injection is not the two-product source scaling plus fixed root action")
for function in (
    "relative_hermitian_trace",
    "fixed_root_injection",
    "apply_shear",
    "project_final_boundary",
):
    forbidden = {"split_to_full", "ring_multiply"}.intersection(calls(production, function))
    if forbidden:
        raise SystemExit(f"production accepted path materializes full cyclotomic work: {function} {forbidden}")
for function in ("ring_multiply", "ring_conjugate", "execute_dual"):
    if function not in {node.name for node in ast.walk(ast.parse(Path(oracle).read_text())) if isinstance(node, ast.FunctionDef)}:
        raise SystemExit(f"oracle lacks independent semantic function {function}")
PY

"$python" -m py_compile "$source_path" "$oracle_path"
nice -n 10 "$python" -X dev "$source_path" \
  >"$result" 2>"$evidence_dir/m117.qualifier.result.stderr"
test ! -s "$evidence_dir/m117.qualifier.result.stderr"
nice -n 10 "$python" -X dev "$oracle_path" \
  >"$oracle_result" 2>"$evidence_dir/m117.qualifier.oracle.stderr"
test ! -s "$evidence_dir/m117.qualifier.oracle.stderr"

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
  and .logical_carrier_cells == 17
  and .logical_integer_coordinates == 272
  and (.intermediate_phase_cells_projected|not)
  and .full_cyclotomic_carrier_materializations == 0
  and .split_to_full_scalar_lifts == 0
  and .retained_inverse_history_bytes == 0
  and [.cases[].boundary] == [197,112]
  and [.cases[].coupling_disabled_boundary] == [16,-1]
  and [.cases[].accepted_named_component_maxima_sum_bits] == [5177,5123]
  and all(.cases[];
    .restored_exactly
    and .same_backing
    and .stats.relative_hermitian_trace_calls == 7
    and .stats.relative_hermitian_trace_real_multiplications == 21
    and .stats.fixed_root_injection_calls == 6
    and .stats.fixed_root_injection_real_multiplications == 12
    and .stats.fixed_root_action_steps == 28
    and .stats.real_subfield_ring_multiplications == 33
  )
  and all(.algebra_controls[]; .)
  and all(.carrier_controls[]; .)
  and .restoration_reuse_case.primary_restored_exactly
  and .restoration_reuse_case.reuse_restored_exactly
  and .restoration_reuse_case.same_original_backing
  and .restoration_reuse_case.fresh_restored_reuse_boundary_equal
  and .restoration_reuse_case.fresh_restored_reuse_rank_and_arithmetic_signature_equal
  and (.restoration_reuse_case.fresh_restored_reuse_full_metadata_sensitive_signature_equal|not)
  and .restoration_reuse_case.restored_minus_fresh_resident_payload_bits == 2
  and (.restoration_reuse_case.baseline_reload|not)
  and .relative_phase_sensitivity_diagnostic.coupling_changes_tested_boundary
  and (.relative_phase_sensitivity_diagnostic.executed_physical_dephasing_model|not)
  and (.relative_phase_sensitivity_diagnostic.relative_phase_sensitivity_is_distinct_resource|not)
  and .matched_classical.equal_or_lower_resource_signature_available
  and (.matched_classical.comparison_establishes_advantage|not)
  and (.resource_law.exact_real_multiply_internal_accumulator_scratch_bounded|not)
  and (.resource_law.named_component_maxima_sum_is_complete_material_peak|not)
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.not_established|index("COMPUTATIONAL_ADVANTAGE")) != null
  and (.not_established|index("SMALL_WALL_CROSSING")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and (.production_module_imported|not)
  and .semantic_oracle == "INDEPENDENT_CANONICAL_DEGREE16_CYCLOTOMIC_POWER_BASIS"
  and [.cases[].boundary] == [197,112]
  and [.cases[].coupling_disabled_boundary] == [16,-1]
  and [.cases[].single_site_perturbed_boundary] == [-194,61]
  and [.cases[].alternate_descriptor_boundary] == [27,10]
  and all(.cases[];
    . as $case
    |
    .execution.full_pair_boundary_equal
    and .execution.all_forward_inverse_step_states_equal
    and .execution.full_restored_exactly
    and .execution.pair_restored_exactly
    and .perturbed_execution.full_pair_boundary_equal
    and .perturbed_execution.full_restored_exactly
    and .alternate_execution.full_pair_boundary_equal
    and .alternate_execution.full_restored_exactly
    and all($case.global_phase_rotated_boundaries[]; . == $case.boundary)
    and .same_order_inverse_fails
    and .wrong_arithmetic_inverse_fails
    and .all_three_shear_pairs_noncommute
  )
  and .observed_resource_law.accepted_pair_relative_trace_calls == 7
  and .observed_resource_law.accepted_pair_relative_trace_real_multiplications == 21
  and .observed_resource_law.accepted_pair_injection_calls == 6
  and .observed_resource_law.accepted_pair_injection_real_multiplications == 12
  and .observed_resource_law.accepted_pair_root_action_steps == 28
  and .observed_resource_law.strongest_compact_reference_is_identical_to_production_recurrence
  and (.not_established|index("DISTINCT_PHASE_RESOURCE")) != null
' "$oracle_result" >/dev/null

"$python" - "$result" "$oracle_result" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    production = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    oracle = json.load(handle)

production_cases = {case["family"]: case for case in production["cases"]}
oracle_cases = {case["family"]: case for case in oracle["cases"]}
for family in ("PRIMARY", "REUSE"):
    if production_cases[family]["boundary"] != oracle_cases[family]["boundary"]:
        raise SystemExit(f"boundary mismatch for {family}")
    pstats = production_cases[family]["stats"]
    ostats = oracle_cases[family]["execution"]["pair_stats"]
    checks = {
        "real_subfield_ring_multiplications": "real_subfield_ring_multiplications",
        "real_subfield_coefficient_multiplications": "real_subfield_coefficient_multiplications",
        "relative_hermitian_trace_calls": "relative_trace_calls",
        "fixed_root_injection_calls": "injection_calls",
        "fixed_root_action_steps": "root_action_steps",
    }
    for production_name, oracle_name in checks.items():
        if pstats[production_name] != ostats[oracle_name]:
            raise SystemExit(
                f"resource mismatch for {family}: {production_name}/{oracle_name}"
            )
PY

echo "QUALIFIED_F17_THREE_SHEAR_RELATIVE_HERMITIAN_TRACE_FEEDBACK"
