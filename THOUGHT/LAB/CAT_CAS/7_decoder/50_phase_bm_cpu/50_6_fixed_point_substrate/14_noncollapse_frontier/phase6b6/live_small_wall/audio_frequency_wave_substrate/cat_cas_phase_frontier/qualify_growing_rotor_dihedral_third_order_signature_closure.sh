#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_dihedral_third_order_signature_closure.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_dihedral_third_order_signature_closure.py"
oracle="$here/growing_rotor_dihedral_third_order_signature_closure_independent_oracle.py"
sealed="$here/GROWING_ROTOR_DIHEDRAL_THIRD_ORDER_SIGNATURE_CLOSURE_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_DIHEDRAL_THIRD_ORDER_SIGNATURE_CLOSURE_INDEPENDENT_ORACLE.json"

mkdir -p "$build_dir" "$build_dir/pycache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$build_dir/pycache"

nice -n 10 ionice -c 3 "$python" "$production" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("third-order production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("third-order oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_REFINED_CLOSURE" and
  .selected_triangle_coordinates == [[1,2,3],[1,4,5]] and
  .minimality.observed_triangle_coordinate_candidates == 33 and
  .minimality.homometric_bracelet_pairs_requiring_separation == 24 and
  .minimality.successful_single_coordinates == 0 and
  .minimality.successful_coordinate_pairs == 8 and
  .minimality.minimal_selected_coordinate_count_within_candidate_family == 2 and
  ([.topology_cases[].rotors] == [2,3,4,5,6]) and
  ([.topology_cases[].necklace_cells] == [9,57,285,1197,4389]) and
  ([.topology_cases[].bracelet_cells] == [9,33,165,621,2277]) and
  ([.topology_cases[].refined_signature_cells] == [9,33,165,621,2277]) and
  ([.topology_cases[].prior_materialized_plan_nonzeros] == [272,2448,21904,131168,652048]) and
  ([.topology_cases[].refined_signatures_equal_dihedral_orbits] | all) and
  ([.topology_cases[].all_sixteen_shift_bases_equitable] | all) and
  ([.transaction_cases[].prime] == [103,239]) and
  ([.transaction_cases[].retained_shift_basis_plans] | min == 16) and
  ([.transaction_cases[].retained_shift_plan_nonzero_entries] | min == 652048) and
  ([.transaction_cases[].public_signature_descriptor_integer_cells] | min == 25047) and
  ([.transaction_cases[].public_representative_descriptor_integer_cells] | min == 38709) and
  ([.transaction_cases[].public_plan_compiler_third_order_triangle_shape_tests] | min == 13692480) and
  ([.transaction_cases[].primary_restoration_error_field_cells] | max == 0) and
  ([.transaction_cases[].reuse_restoration_error_field_cells] | max == 0) and
  ([.transaction_cases[].same_backing_primary] | all) and
  ([.transaction_cases[].same_backing_reuse] | all) and
  ([.transaction_cases[].restoration_generation_after_reuse] | min == 2) and
  ([.transaction_cases[].controls.missing_inverse_error_field_cells] | min > 0) and
  ([.transaction_cases[].controls.wrong_inverse_error_field_cells] | min > 0) and
  ([.transaction_cases[].controls.reordered_inverse_error_field_cells] | min > 0) and
  ([.transaction_cases[].controls.null_carrier_rejected] | all) and
  .matched_classical_recurrence == "IDENTICAL_PLAN_COMPILED_REFINED_SIGNATURE_QUOTIENT_RECURRENCE" and
  (.full_bracelet_fallback_required | not) and
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
  .result == "PASS_REFINED_CLOSURE" and
  (.production_source_imported | not) and
  (.production_transition_called | not) and
  .multiset_occupation_oracle and
  .ordered_particle_transition_oracle and
  ([.topology_cases[].refined_signature_cells] == [9,33,165,621,2277]) and
  ([.transaction_cases[].prime] == [103,239])
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
    "selected_triangle_coordinates",
    "minimality",
    "topology_cases",
    "transaction_cases",
    "matched_classical_recurrence",
    "full_bracelet_fallback_required",
    "refinement_smaller_than_full_bracelet_identity",
    "distinct_phase_resource_established",
    "computational_advantage",
    "small_wall_crossed",
    "terminal",
):
    if production[key] != oracle[key]:
        raise SystemExit(f"independent third-order field differs: {key}")
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
        "growing_rotor_dihedral_third_order_signature_closure" in name
        or "growing_rotor_pair_signature_streamed_quotient" in name
        for name in names
    ):
        raise SystemExit("third-order oracle production dependency isolation failed")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_DIHEDRAL_THIRD_ORDER_SIGNATURE_CLOSURE_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
