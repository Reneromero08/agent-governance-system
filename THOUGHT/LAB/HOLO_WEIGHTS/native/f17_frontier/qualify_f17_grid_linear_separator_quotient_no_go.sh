#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export PYTHONHASHSEED=0

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_f17_grid_linear_separator_quotient_no_go.sh EVIDENCE_DIR" >&2
  exit 2
fi

evidence_dir=$1
frontier_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=$(git -C "$frontier_dir" rev-parse --show-toplevel)
python="$repo/.venv/bin/python"
source_path="$frontier_dir/f17_grid_linear_separator_quotient_no_go.py"
oracle_path="$frontier_dir/f17_grid_linear_separator_quotient_no_go_oracle.py"
m120_path="$frontier_dir/f17_runtime_weighted_grid_kronecker_butterfly_closure.py"
m119_path="$frontier_dir/f17_runtime_weighted_grid_phase_factor_closure.py"
expected_path="$frontier_dir/F17_GRID_LINEAR_SEPARATOR_QUOTIENT_NO_GO_RESULTS.json"
oracle_expected_path="$frontier_dir/F17_GRID_LINEAR_SEPARATOR_QUOTIENT_NO_GO_ORACLE_RESULTS.json"
provenance_path="$frontier_dir/F17_GRID_LINEAR_SEPARATOR_QUOTIENT_NO_GO_PROVENANCE.json"
review_path="$frontier_dir/F17_GRID_LINEAR_SEPARATOR_QUOTIENT_NO_GO_INDEPENDENT_REVIEW.md"
qualifier_path="$frontier_dir/qualify_f17_grid_linear_separator_quotient_no_go.sh"
result="$evidence_dir/m121.qualifier.result.full.json"
oracle_result="$evidence_dir/m121.qualifier.oracle.full.json"

mkdir -p "$evidence_dir"
export PYTHONPYCACHEPREFIX="$evidence_dir/pycache"

for tool in cmp git jq nice rg sha256sum; do
  command -v "$tool" >/dev/null
done
test -x "$python"
jq empty "$expected_path" "$oracle_expected_path" "$provenance_path"

scientific_parent=$(jq -r '.scientific_source_parent' "$provenance_path")
test "$scientific_parent" = "f279ab62874fe1a1e0331c99fcf94430f0df3512"
git -C "$repo" cat-file -e "$scientific_parent^{commit}"
git -C "$repo" merge-base --is-ancestor "$scientific_parent" HEAD

for sealed_path in \
  "$source_path" \
  "$oracle_path" \
  "$m120_path" \
  "$m119_path" \
  "$qualifier_path" \
  "$expected_path" \
  "$oracle_expected_path" \
  "$review_path"
do
  sealed_name=$(basename "$sealed_path")
  sealed_expected=$(jq -r --arg name "$sealed_name" '.files[$name] // empty' "$provenance_path")
  test -n "$sealed_expected"
  test "$(sha256sum "$sealed_path" | cut -d' ' -f1)" = "$sealed_expected"
done

if rg -n '^[[:space:]]*(from|import)[[:space:]]+(f17_grid_linear_separator_quotient_no_go|f17_runtime_weighted_grid_kronecker_butterfly_closure|f17_runtime_weighted_grid_phase_factor_closure)([[:space:]]|$)' "$oracle_path"; then
  echo "independent M121 oracle imports production or its phase backend" >&2
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

accepted = function_node(production_tree, "execute_descriptor_transaction")
call_nodes = sorted(
    (child for child in ast.walk(accepted) if isinstance(child, ast.Call)),
    key=lambda child: (child.lineno, child.col_offset),
)
called = [
    child.func.attr if isinstance(child.func, ast.Attribute) else (
        child.func.id if isinstance(child.func, ast.Name) else ""
    )
    for child in call_nodes
]
required_order = ["load_factor_seed", "apply_operation", "project_boundary", "apply_operation", "unload_factor_seed"]
positions = []
cursor = 0
for required in required_order:
    try:
        cursor = called.index(required, cursor)
    except ValueError as exc:
        raise SystemExit(f"accepted transaction lacks ordered call {required}") from exc
    positions.append(cursor)
    cursor += 1
if positions != sorted(positions):
    raise SystemExit("accepted transaction call order changed")

certificate = function_node(production_tree, "analytic_linear_separator_certificate")
for child in ast.walk(certificate):
    if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        for generator in child.generators:
            if isinstance(generator.iter, ast.Call) and isinstance(generator.iter.func, ast.Name) and generator.iter.func.id == "range":
                if any(isinstance(argument, ast.Name) and argument.id == "width" for argument in generator.iter.args):
                    raise SystemExit("accepted formula certificate enumerates the separator width")
    if isinstance(child, ast.Call):
        name = child.func.attr if isinstance(child.func, ast.Attribute) else (
            child.func.id if isinstance(child.func, ast.Name) else ""
        )
        if name in {"rank_mod", "matrix_product", "explicit_rank_certificate"}:
            raise SystemExit("accepted formula certificate calls dense verification machinery")

required_oracle = {
    "butterfly_boundary",
    "gray_histogram",
    "independent_factor_restoration",
    "explicit_rank_certificate",
    "rank_mod",
    "matrix_product",
}
present = {node.name for node in ast.walk(oracle_tree) if isinstance(node, ast.FunctionDef)}
missing = required_oracle - present
if missing:
    raise SystemExit(f"independent M121 oracle lacks {sorted(missing)}")
PY

"$python" -m py_compile "$source_path" "$oracle_path"
nice -n 10 "$python" -X dev "$source_path" >"$result" 2>"$evidence_dir/m121.qualifier.result.stderr"
test ! -s "$evidence_dir/m121.qualifier.result.stderr"
nice -n 10 "$python" -X dev "$oracle_path" >"$oracle_result" 2>"$evidence_dir/m121.qualifier.oracle.stderr"
test ! -s "$evidence_dir/m121.qualifier.oracle.stderr"

cmp "$result" "$expected_path"
cmp "$oracle_result" "$oracle_expected_path"
test "$(sha256sum "$result" | cut -d' ' -f1)" = "$(jq -r '.reference_full_result_sha256' "$provenance_path")"
test "$(sha256sum "$oracle_result" | cut -d' ' -f1)" = "$(jq -r '.reference_full_oracle_sha256' "$provenance_path")"

jq -e '
  .result == "PASS_UNIFORM_EXACT_LINEAR_SEPARATOR_QUOTIENT_OBSTRUCTION"
  and .classification_candidate == "SOURCE_AUDITED_PACKAGE_LOCAL"
  and .verification_level_candidate == "PACKAGE_SELF_REVIEW"
  and .factor_carrier_restoration_class == "EXACT_ALGEBRAIC_RESTORATION"
  and .transient_projection_buffer_restoration_class == "NO_RESTORATION_CLAIM"
  and .runtime_interface_advance == "ARBITRARY_PUBLIC_NONZERO_F17_UNARY_AND_EDGE_DESCRIPTORS_ON_COMPILED_N2_N3_N4_GRID_TOPOLOGY"
  and (.accepted_path_continuation_family_enumerated|not)
  and (.accepted_path_dense_rank_matrix_materialized|not)
  and .accepted_path_final_full_lifts_per_transaction == 1
  and (.intermediate_factor_or_separator_values_projected|not)
  and .retained_inverse_history_bytes == 0
  and (.cases|length) == 6
  and all(.cases[];
    .boundary_agreement
    and .restored_exactly
    and .same_backing
    and .linear_separator_certificate.continuation_tensor_rank == pow(2; .n)
    and .linear_separator_certificate.vertical_tensor_rank == pow(2; .n)
    and .linear_separator_certificate.combined_uniform_legal_continuation_observation_rank == pow(2; .n)
    and .linear_separator_certificate.uniform_exact_linear_separator_minimum_field_coordinates == pow(2; .n)
    and (.linear_separator_certificate.dense_width_by_width_matrix_materialized_by_accepted_certificate|not)
    and .resources.accepted_path_dense_rank_matrix_cells == 0
    and .resources.accepted_path_continuation_family_enumerations == 0
    and .resources.transient_projection_buffer_restoration_class == "NO_RESTORATION_CLAIM"
  )
  and all(.restoration_reuse[];
    .same_original_backing
    and .fresh_restored_reuse_signature_equal
    and .canonical_restored_state.all_factor_cells_zero
    and (.baseline_reload|not)
    and .retained_inverse_history_bytes == 0
  )
  and .controls.false_linear_rank_cap_rejected
  and .controls.exact_linear_rank_cap_accepted
  and .controls.zero_unary_descriptor_rejected
  and .controls.zero_edge_descriptor_rejected
  and .controls.arbitrary_legal_descriptor_mutation_changes_boundary
  and .controls.coordinate_drop_rank7_has_nonzero_kernel_vector
  and .controls.valid_continuation_separates_dropped_last_coordinate
  and .controls.duplicate_local_continuation_choices_make_local_rank_one
  and .controls.zero_vertical_weight_would_make_local_rank_one_and_halve_tensor_rank
  and (.controls.compiled_public_topology_contains_runtime_weights|not)
  and .controls.missing_inverse_leaves_resident_state
  and .controls.wrong_inverse_exponent_fails_restoration
  and .controls.reordered_noncommuting_inverse_fails
  and .controls.resident_mutation_detected
  and .controls.snapshot_reload_absent
  and (.formula_certificates_n1_through_n16|length) == 16
  and all(.formula_certificates_n1_through_n16[];
    .continuation_tensor_rank == pow(2; .separator_binary_width)
    and .vertical_tensor_rank == pow(2; .separator_binary_width)
    and .combined_uniform_legal_continuation_observation_rank == pow(2; .separator_binary_width)
    and (.dense_width_by_width_matrix_materialized_by_accepted_certificate|not)
  )
  and .theorem.uniform_interface_scope == "FIXED_LINEAR_ENCODER_MUST_SUPPORT_ARBITRARY_FIELD_MESSAGES_AND_EVERY_LEGAL_NONZERO_RUNTIME_CONTINUATION"
  and .theorem.uniform_exact_linear_quotient_below_two_to_the_n == "REJECTED"
  and .theorem.program_dependent_or_nonlinear_quotient == "NOT_ADJUDICATED"
  and .matched_classical.strongest_evaluated_operational_recurrence == "IDENTICAL_EXACT_KRONECKER_BUTTERFLY_ON_TWO_TO_THE_N_Q_ZETA17_MESSAGES"
  and (.matched_classical.all_order_add_mtbdd_mps_matchgate_or_boundary_specific_algorithms_exhausted|not)
  and (.matched_classical.comparison_establishes_advantage|not)
  and (.rejected_interpretations|index("DISTINCT_PHASE_RESOURCE")) != null
  and (.rejected_interpretations|index("COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING")) != null
  and (.rejected_interpretations|index("REPLACEMENT_OF_PHYSICAL_BITS_WITH_PI")) != null
  and (.terminal|not)
' "$result" >/dev/null

jq -e '
  .result == "PASS"
  and (.imports_production_m121_m120_or_m119|not)
  and (.imports_phase_backend|not)
  and .finite_field_rank_checks == [103,137]
  and .all_boundaries_reconstructed_by_two_independent_exact_recurrences
  and .all_factor_cells_restore_exactly_on_same_backing
  and .all_continuation_vertical_and_combined_ranks_are_full
  and .all_rank_halving_controls_pass
  and .coordinate_drop_counterexample_control_passes
  and (.cases|length) == 6
  and all(.cases[];
    .butterfly_gray_boundary_agreement
    and .descriptor_mutation_changes_boundary
    and .factor_restoration.forward_changed_seed
    and .factor_restoration.seed_restored_exactly
    and .factor_restoration.unload_restored_zero_backing
    and .factor_restoration.same_backing
    and all(.rank_certificates[];
      .continuation_rank == .width
      and .vertical_rank == .width
      and .combined_rank == .width
      and .zero_vertical_weight_rank == (.width / 2)
      and .duplicate_local_continuation_choice_rank == (.width / 2)
      and .coordinate_drop_encoder_rank == (.width - 1)
      and .coordinate_drop_kernel_is_nonzero
      and .valid_continuation_sees_dropped_kernel
    )
  )
' "$oracle_result" >/dev/null

"$python" - "$result" "$oracle_result" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    production = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    oracle = json.load(handle)
for actual, reference in zip(production["cases"], oracle["cases"], strict=True):
    for key in ("n", "family", "plan_fingerprint", "unary_weights", "edge_weights"):
        if actual[key] != reference[key]:
            raise SystemExit(f"M121 production/oracle {key} mismatch")
    if actual["boundary"] != reference["canonical_boundary"]:
        raise SystemExit("M121 production/oracle exact boundary mismatch")
    width = 1 << actual["n"]
    for certificate in reference["rank_certificates"]:
        if certificate["combined_rank"] != width:
            raise SystemExit("M121 independent combined rank is not full")
PY

echo "QUALIFIED_F17_GRID_LINEAR_SEPARATOR_QUOTIENT_NO_GO"
