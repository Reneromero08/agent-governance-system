#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_pair_scattering_krylov_diagnostic.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_pair_scattering_krylov_diagnostic.cpp"
oracle="$here/growing_rotor_pair_scattering_krylov_independent_oracle.py"
sealed="$here/GROWING_ROTOR_PAIR_SCATTERING_KRYLOV_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_PAIR_SCATTERING_KRYLOV_INDEPENDENT_ORACLE.json"
binary="$build_dir/growing_rotor_pair_scattering_krylov_diagnostic"

mkdir -p "$build_dir"
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
  "$production" -o "$binary"

nice -n 10 ionice -c 3 "$binary" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("growing-rotor production reexecution differs from seal")
' "$sealed"

export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX
nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("growing-rotor oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION" and
  .result == "PASS" and
  ([.cases[].rotors] == [2,3,4,5]) and
  ([.cases[].necklace_cells] == [9,57,285,1197]) and
  ([.cases[].occupation_histograms_visited_during_topology_compile] == [153,969,4845,20349]) and
  ([.cases[].labelled_cells] == [289,4913,83521,1419857]) and
  ([.cases[].f103_krylov_degree] == [9,57,284,1182]) and
  ([.cases[].f137_krylov_degree] == [9,57,284,1190]) and
  ([.cases[].f103_dimension_deficit] == [0,0,1,15]) and
  ([.cases[].f137_dimension_deficit] == [0,0,1,7]) and
  ([.cases[].primary_restoration_error] | max < 2e-10) and
  ([.cases[].reuse_restoration_error] | max < 2e-10) and
  ([.cases[].fresh_restored_reuse_boundary_error] | max < 2e-10) and
  .controls.missing_inverse_error > 1e-6 and
  .controls.wrong_inverse_error > 1e-6 and
  .controls.reordered_inverse_error > 1e-6 and
  .controls.repeated_reuse_cycles == 16 and
  .controls.repeated_reuse_max_error < 8e-10 and
  .controls.null_carrier_rejected and
  .resource_law.inverse_plan_rematerialized_from_public_topology and
  (.resource_law.compiled_rows_retained_for_inverse | not) and
  .resource_law.retained_inverse_history_bytes == 0 and
  .resource_law.accepted_dense_operator_cells == 0 and
  .resource_law.accepted_occupation_vector_cells == 0 and
  (.stable_transferable_recurrence_quotient_established | not) and
  .matched_classical_recurrence == "IDENTICAL_GROWING_NECKLACE_PAIR_SCATTERING_AND_DIAGONAL_PHASE_RECURRENCE" and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.catvm_custody | not) and
  (.physical_waveform_execution | not) and
  (.physical_bit_replacement | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NUMERICAL_PHYSICAL_STATE_RESTORATION" and
  .result == "PASS" and
  ([.cases[].rotors] == [2,3,4,5]) and
  ([.cases[].full_occupation_cells] == [153,969,4845,20349]) and
  ([.cases[].necklace_cells] == [9,57,285,1197]) and
  ([.cases[].f103_krylov_degree] == [9,57,284,1182]) and
  ([.cases[].f137_krylov_degree] == [9,57,284,1190]) and
  ([.cases[].weighted_hermitian_residual] | max == 0) and
  .controls.missing_inverse_error > 1e-6 and
  .controls.wrong_inverse_error > 1e-6 and
  .controls.reordered_inverse_error > 1e-6 and
  .controls.repeated_reuse_cycles == 16 and
  .controls.repeated_reuse_max_error < 8e-10 and
  (.production_source_imported | not) and
  (.production_projection_called | not) and
  (.production_inverse_called | not) and
  .oracle_sparse_generators_are_verification_only and
  (.stable_transferable_recurrence_quotient_established | not) and
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
    raise SystemExit("independent growing-rotor case count differs")
exact_keys = (
    "rotors",
    "necklace_cells",
    "labelled_cells",
    "enumerated_generator_terms",
    "weighted_particle_pair_shift_terms",
    "unique_nonzero_generator_terms",
    "f103_krylov_degree",
    "f137_krylov_degree",
    "f103_dimension_deficit",
    "f137_dimension_deficit",
)
for production_case, oracle_case in zip(
    production["cases"], oracle["cases"], strict=True
):
    for key in exact_keys:
        if production_case[key] != oracle_case[key]:
            raise SystemExit(f"independent exact case metric differs: {key}")
    boundary_error = max(
        abs(left - right)
        for left, right in zip(
            production_case["primary_boundary"],
            oracle_case["primary_boundary"],
            strict=True,
        )
    )
    if boundary_error > 2e-12:
        raise SystemExit("independent growing-rotor boundary differs")
for key in (
    "missing_inverse_error",
    "wrong_inverse_error",
    "reordered_inverse_error",
):
    if abs(production["controls"][key] - oracle["controls"][key]) > 2e-12:
        raise SystemExit(f"independent growing-rotor control differs: {key}")
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
    if any("growing_rotor" in name or "necklace_offdiagonal" in name for name in names):
        raise SystemExit("growing-rotor oracle production dependency isolation failed")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_PAIR_SCATTERING_KRYLOV_DIAGNOSTIC_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
