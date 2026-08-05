#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_pair_signature_fiber_quotient.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_pair_signature_fiber_quotient.py"
oracle="$here/growing_rotor_pair_signature_fiber_quotient_independent_oracle.py"
sealed="$here/GROWING_ROTOR_PAIR_SIGNATURE_FIBER_QUOTIENT_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_PAIR_SIGNATURE_FIBER_QUOTIENT_INDEPENDENT_ORACLE.json"

mkdir -p "$build_dir"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

nice -n 10 ionice -c 3 "$python" "$production" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("fiber quotient production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("fiber quotient oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS" and
  .quotient_cell_law == [9,33,165,621] and
  .necklace_cell_law == [9,57,285,1197] and
  .all_sixteen_shift_bases_equitable and
  .pair_diagonal_closed_by_signature and
  ([.cases[].rotors] == [2,2,3,3,4,4,5,5]) and
  ([.cases[].prime] == [103,239,103,239,103,239,103,239]) and
  ([.cases[].pair_signature_fiber_cells] == [9,9,33,33,165,165,621,621]) and
  (.cases | all(.primary_boundary == .full_necklace_primary_boundary)) and
  ([.cases[].primary_restoration_error_field_cells] | max == 0) and
  ([.cases[].reuse_restoration_error_field_cells] | max == 0) and
  ([.cases[].same_backing_primary] | all) and
  ([.cases[].same_backing_reuse] | all) and
  ([.cases[].restoration_generation_after_reuse] | min == 2) and
  ([.cases[].controls.missing_inverse_error_field_cells] | min > 0) and
  ([.cases[].controls.wrong_inverse_error_field_cells] | min > 0) and
  ([.cases[].controls.reordered_inverse_error_field_cells] | min > 0) and
  ([.cases[].controls.null_carrier_rejected] | all) and
  .resource_law.accepted_public_shift_basis_plans == 16 and
  .resource_law.accepted_retained_inverse_history_bytes == 0 and
  .resource_law.accepted_assignment_or_relation_table_cells == 0 and
  .matched_classical_recurrence == "IDENTICAL_PAIR_SIGNATURE_FIBER_QUOTIENT_RECURRENCE" and
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
  .result == "PASS" and
  .quotient_cell_law == [9,33,165,621] and
  .necklace_cell_law == [9,57,285,1197] and
  .all_sixteen_shift_bases_equitable and
  (.production_source_imported | not) and
  (.production_projection_called | not) and
  (.production_inverse_called | not) and
  .full_necklace_shift_bases_are_verification_only and
  .matched_classical_recurrence == "IDENTICAL_PAIR_SIGNATURE_FIBER_QUOTIENT_RECURRENCE" and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
if len(production["cases"]) != len(oracle["cases"]):
    raise SystemExit("independent fiber quotient case count differs")
exact_keys = (
    "rotors",
    "prime",
    "multiplicative_generator",
    "seventeenth_root",
    "necklace_cells",
    "pair_signature_fiber_cells",
    "occupation_histograms_visited",
    "primary_depth",
    "reuse_depth",
    "primary_boundary",
    "full_necklace_primary_boundary",
    "reuse_boundary",
    "fresh_reuse_boundary",
    "primary_output_commitment",
    "full_necklace_output_commitment",
    "primary_restoration_error_field_cells",
    "reuse_restoration_error_field_cells",
    "restoration_generation_after_reuse",
    "enumerated_mode_pair_shift_terms",
    "weighted_particle_pair_shift_terms",
    "quotient_shift_plan_nonzero_entries",
    "equitability_comparisons",
)
for production_case, oracle_case in zip(
    production["cases"], oracle["cases"], strict=True
):
    for key in exact_keys:
        if production_case[key] != oracle_case[key]:
            raise SystemExit(f"independent exact fiber metric differs: {key}")
    if production_case["controls"] != oracle_case["controls"]:
        raise SystemExit("independent exact fiber controls differ")
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
    if any("growing_rotor_pair_signature" in name for name in names):
        raise SystemExit("fiber quotient oracle production dependency isolation failed")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_PAIR_SIGNATURE_FIBER_QUOTIENT_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
