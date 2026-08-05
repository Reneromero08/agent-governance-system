#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_jastrow_phase_chart_diagnostic.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_jastrow_phase_chart_diagnostic.py"
oracle="$here/growing_rotor_jastrow_phase_chart_independent_oracle.py"
sealed="$here/GROWING_ROTOR_JASTROW_PHASE_CHART_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_JASTROW_PHASE_CHART_INDEPENDENT_ORACLE.json"

mkdir -p "$build_dir"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

nice -n 10 ionice -c 3 "$python" "$production" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("Jastrow production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("Jastrow independent reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS" and
  ([.cases[].rotors] == [2,2,3,3,4,4,5,5]) and
  ([.cases[].prime] == [103,239,103,239,103,239,103,239]) and
  ([.cases[].necklace_cells] == [9,9,57,57,285,285,1197,1197]) and
  ([.cases[].occupation_histograms_visited] == [153,153,969,969,4845,4845,20349,20349]) and
  ([.cases[].source_chart_membership.closed] | all) and
  ([.cases[].diagonal_only_chart_membership.closed] | all) and
  ([.cases[] | select(.rotors == 2) | .primary_scattering_chart_membership.closed] | all) and
  ([.cases[] | select(.rotors == 2) | .alternate_scattering_chart_membership.closed] | all) and
  ([.cases[] | select(.rotors >= 3) | (.primary_scattering_chart_membership.closed | not)] | all) and
  ([.cases[] | select(.rotors >= 3) | (.alternate_scattering_chart_membership.closed | not)] | all) and
  .first_primary_escape_rotor_count == 3 and
  .two_rotor_chart_closure_is_dimension_saturation_not_transfer and
  ([.cases[].primary_restoration_error_field_cells] | max == 0) and
  ([.cases[].reuse_restoration_error_field_cells] | max == 0) and
  ([.cases[].same_backing_primary] | all) and
  ([.cases[].same_backing_reuse] | all) and
  ([.cases[].restoration_generation_after_reuse] | min == 2) and
  ([.cases[].controls.missing_inverse_error_field_cells] | min > 0) and
  ([.cases[].controls.wrong_inverse_error_field_cells] | min > 0) and
  ([.cases[].controls.reordered_inverse_error_field_cells] | min > 0) and
  ([.cases[].controls.null_carrier_rejected] | all) and
  .resource_law.accepted_retained_inverse_history_bytes == 0 and
  .resource_law.inverse_word_rematerialized_from_public_topology and
  .resource_law.accepted_assignment_or_relation_table_cells == 0 and
  .matched_classical_recurrence == "IDENTICAL_FULL_NECKLACE_EXACT_K_TIMES_D_SHEAR_RECURRENCE" and
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
  .first_escape_rotor_count == 3 and
  .two_rotor_chart_closure_is_dimension_saturation_not_transfer and
  (.production_source_imported | not) and
  (.production_membership_called | not) and
  (.production_projection_called | not) and
  (.production_inverse_called | not) and
  .oracle_equation_matrices_are_verification_only and
  (.compact_nine_parameter_chart_survives_growing_offdiagonal_scattering | not) and
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
    raise SystemExit("independent Jastrow case count differs")
exact_keys = (
    "rotors",
    "prime",
    "multiplicative_generator",
    "seventeenth_root",
    "necklace_cells",
    "occupation_histograms_visited",
    "primary_output_commitment",
    "alternate_output_commitment",
    "primary_boundary",
    "reuse_boundary",
    "fresh_reuse_boundary",
    "primary_restoration_error_field_cells",
    "reuse_restoration_error_field_cells",
    "restoration_generation_after_reuse",
    "generator_unique_terms",
)
for production_case, oracle_case in zip(
    production["cases"], oracle["cases"], strict=True
):
    for key in exact_keys:
        if production_case[key] != oracle_case[key]:
            raise SystemExit(f"independent exact Jastrow metric differs: {key}")
    for key in (
        "source_chart_membership",
        "diagonal_only_chart_membership",
        "primary_scattering_chart_membership",
        "alternate_scattering_chart_membership",
    ):
        if production_case[key]["closed"] != oracle_case[key]["closed"]:
            raise SystemExit(f"independent Jastrow membership differs: {key}")
    if production_case["controls"] != oracle_case["controls"]:
        raise SystemExit("independent exact Jastrow controls differ")
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
    if any("growing_rotor_jastrow" in name for name in names):
        raise SystemExit("Jastrow oracle production dependency isolation failed")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_JASTROW_PHASE_CHART_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
