#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_dihedral_third_order_signature_streamed_closure.sh MANAGED_BUILD_DIR" >&2
  exit 2
fi

build_dir=$1
case "$build_dir" in
  /dev/shm|/dev/shm/*|/run/shm|/run/shm/*)
    echo "RAM-backed build directories are forbidden" >&2
    exit 2
    ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
python="$repo_root/.venv/bin/python"
production="$here/growing_rotor_dihedral_third_order_signature_streamed_closure.py"
oracle="$here/growing_rotor_dihedral_third_order_signature_streamed_closure_independent_oracle.py"
sealed="$here/GROWING_ROTOR_DIHEDRAL_THIRD_ORDER_SIGNATURE_STREAMED_CLOSURE_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_DIHEDRAL_THIRD_ORDER_SIGNATURE_STREAMED_CLOSURE_INDEPENDENT_ORACLE.json"

mkdir -p "$build_dir" "$build_dir/pycache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$build_dir/pycache"

nice -n 10 ionice -c 3 "$python" "$production" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("streamed third-order production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("streamed third-order oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_ZERO_RETAINED_SHIFT_PLAN_WITH_MEASURED_REMATERIALIZATION" and
  .selected_triangle_coordinates == [[1,2,3],[1,4,5]] and
  .public_triangle_stencils == [[[1,3],[2,3]],[[1,5],[4,5]]] and
  ([.topology_cases[].rotors] == [2,3,4,5,6]) and
  ([.topology_cases[].occupation_histograms] == [153,969,4845,20349,74613]) and
  ([.topology_cases[].necklace_cells] == [9,57,285,1197,4389]) and
  ([.topology_cases[].bracelet_cells] == [9,33,165,621,2277]) and
  ([.topology_cases[].refined_signature_cells] == [9,33,165,621,2277]) and
  ([.topology_cases[].prior_materialized_plan_nonzeros] == [272,2448,21904,131168,652048]) and
  ([.topology_cases[].refined_signatures_equal_dihedral_orbits] | all) and
  .transition_audit.mode_pair_shift_terms == 684624 and
  .transition_audit.weighted_particle_shift_terms == 1092960 and
  .transition_audit.analytic_triangle_monomial_delta_evaluations == 24767280 and
  .transition_audit.peak_analytic_triangle_monomials_per_term == 44 and
  .transition_audit.signature_binary_search_comparisons == 7669494 and
  .transition_audit.active_mode_count_histogram == {"0":40272,"3":80544,"4":563808} and
  .transition_audit.analytic_destinations_equal_direct_cyclic_monomial_reexecution and
  .transaction_case.prime == 103 and
  .transaction_case.refined_signature_cells == 2277 and
  .transaction_case.primary_depth == 1 and
  .transaction_case.reuse_depth == 1 and
  .transaction_case.retained_shift_basis_plans == 0 and
  .transaction_case.retained_shift_plan_nonzero_entries == 0 and
  .transaction_case.prior_materialized_plan_nonzero_entries == 652048 and
  .transaction_case.public_signature_descriptor_integer_cells == 25047 and
  .transaction_case.public_representative_descriptor_integer_cells == 38709 and
  .transaction_case.public_triangle_stencil_integer_cells == 8 and
  .transaction_case.primary_restoration_error_field_cells == 0 and
  .transaction_case.reuse_restoration_error_field_cells == 0 and
  .transaction_case.same_backing_primary and
  .transaction_case.same_backing_reuse and
  .transaction_case.restoration_generation_after_reuse == 2 and
  .transaction_case.fresh_restored_reuse_rank_signature_agreement and
  (.transaction_case.baseline_reload_used | not) and
  .transaction_case.controls.missing_inverse_error_field_cells > 0 and
  .transaction_case.controls.wrong_inverse_error_field_cells > 0 and
  .transaction_case.controls.reordered_inverse_error_field_cells > 0 and
  .transaction_case.controls.null_carrier_rejected and
  .transaction_case.primary_boundary == .transaction_case.matched_classical_boundary and
  .transaction_case.primary_output_commitment == .transaction_case.matched_classical_output_commitment and
  .matched_classical_recurrence == "IDENTICAL_TOPOLOGY_STREAMED_TWO_TRIANGLE_REFINED_SIGNATURE_RECURRENCE" and
  .prior_plan_compiled_nonzeros_eliminated == 652048 and
  (.refinement_smaller_than_full_bracelet_identity | not) and
  (.catvm_custody | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.physical_waveform_execution | not) and
  (.physical_bit_replacement | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_ZERO_RETAINED_SHIFT_PLAN_WITH_MEASURED_REMATERIALIZATION" and
  (.production_source_imported | not) and
  (.production_transition_called | not) and
  .multiset_occupation_oracle and
  .explicit_particle_pair_and_triple_oracle and
  .full_anchor_incidence_work_oracle and
  .transition_audit.independent_plan_nonzeros == 652048 and
  .oracle_recurrence == "INDEPENDENT_EXPLICIT_PARTICLE_BRACELET_PLAN"
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
for key in (
    "claim_candidate",
    "claim_ceiling",
    "classification",
    "verification_level",
    "restoration_classification",
    "result",
    "topology_cases",
    "matched_classical_recurrence",
    "refinement_smaller_than_full_bracelet_identity",
):
    if production[key] != oracle[key]:
        raise SystemExit(f"independent streamed third-order field differs: {key}")
for key in (
    "raw_transition_commitment",
    "mode_pair_shift_terms",
    "weighted_particle_shift_terms",
    "analytic_triangle_monomial_delta_evaluations",
    "peak_analytic_triangle_monomials_per_term",
    "active_mode_count_histogram",
):
    if production["transition_audit"][key] != oracle["transition_audit"][key]:
        raise SystemExit(f"independent transition field differs: {key}")
for key in (
    "prime",
    "multiplicative_generator",
    "seventeenth_root",
    "refined_signature_cells",
    "primary_depth",
    "reuse_depth",
    "primary_boundary",
    "reuse_boundary",
    "fresh_reuse_boundary",
    "primary_output_commitment",
    "primary_restoration_error_field_cells",
    "reuse_restoration_error_field_cells",
    "same_backing_primary",
    "same_backing_reuse",
    "restoration_generation_after_reuse",
    "fresh_restored_reuse_rank_signature_agreement",
    "baseline_reload_used",
    "controls",
    "matched_classical_boundary",
    "matched_classical_output_commitment",
):
    if production["transaction_case"][key] != oracle["transaction_case"][key]:
        raise SystemExit(f"independent transaction field differs: {key}")
PY

"$python" - "$oracle" <<'PY'
import ast
import pathlib
import sys

tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        names = [node.module or ""]
    else:
        continue
    if any(
        "growing_rotor_dihedral_third_order_signature_streamed_closure" in name
        or "growing_rotor_pair_signature_streamed_quotient" in name
        for name in names
    ):
        raise SystemExit("streamed third-order oracle dependency isolation failed")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_DIHEDRAL_THIRD_ORDER_SIGNATURE_STREAMED_CLOSURE_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
