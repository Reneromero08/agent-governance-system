#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_dihedral_quotient_bosonic_fourier.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_dihedral_quotient_bosonic_fourier_diagnostic.py"
oracle="$here/growing_rotor_dihedral_quotient_bosonic_fourier_independent_oracle.py"
sealed="$here/GROWING_ROTOR_DIHEDRAL_QUOTIENT_BOSONIC_FOURIER_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_DIHEDRAL_QUOTIENT_BOSONIC_FOURIER_INDEPENDENT_ORACLE.json"

mkdir -p "$build_dir/pycache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$build_dir/pycache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

nice -n 10 ionice -c 3 "$python" "$production" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("dihedral-quotient Fourier production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("dihedral-quotient Fourier oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NO_RESTORATION_CLAIM" and
  .result == "PASS_TESTED_FIXED_QUOTIENT_FOURIER_ROUTES_RETAIN_FORBIDDEN_COST" and
  .quotient_geometry.full_exchange_symmetric_occupation_cells == 74613 and
  .quotient_geometry.momentum_bracelet_cells == 2277 and
  .quotient_geometry.position_zero_total_cells_before_reflection == 4389 and
  .quotient_geometry.position_zero_total_reflection_closed_cells == 2277 and
  .standard_butterfly_route.total_elementary_gates == 577 and
  .standard_butterfly_route.first_forward_gate == ["SHEAR",15,16,10] and
  .standard_butterfly_route.first_gate_orbit_coefficient_mismatch == 60 and
  .standard_butterfly_route.first_gate_rotation_commutator_nonzero_entries == 2 and
  (.standard_butterfly_route.fixed2277_coordinate_quotient_closed_gate_by_gate | not) and
  .direct_kernel_route.full_kernel_cells_if_retained == 5184729 and
  .direct_kernel_route.full_transform_kernel_evaluations_if_streamed == 5184729 and
  .direct_kernel_route.maximum_subset_states_if_every_kernel_is_streamed == 331822656 and
  .direct_kernel_route.dense_row_nonzero_entries == 2277 and
  .direct_kernel_route.dense_column_nonzero_entries == 2277 and
  .direct_kernel_route.dense_column_constant == 17 and
  .direct_kernel_route.generic_permanent_subset_value == 65 and
  .direct_kernel_route.generic_permanent_assignment_value == 65 and
  .direct_kernel_route.wrong_sector_orbit_kernel == 0 and
  (.direct_kernel_route.complete2277_by2277_transform_executed | not) and
  (.direct_kernel_route.universal_structured_transform_lower_bound_established | not) and
  .resource_law.accepted_diagnostic_dense_kernel_cells == 0 and
  .resource_law.accepted_diagnostic_full_occupation_scratch_cells == 0 and
  .resource_law.accepted_diagnostic_retained_transition_plan_entries == 0 and
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
  .restoration_classification == "NO_RESTORATION_CLAIM" and
  (.oracle_imports_cat_cas_modules | not) and
  .topology.occupations == 74613 and
  .topology.momentum_bracelets == 2277 and
  .topology.zero_total_position_occupations == 4389 and
  .topology.reflection_closed_position_cells == 2277 and
  .direct_kernel.dense_row_nonzero_entries == 2277 and
  .direct_kernel.dense_column_nonzero_entries == 2277 and
  .direct_kernel.dense_column_constant == 17 and
  .direct_kernel.generic_ryser_permanent == 65 and
  .direct_kernel.generic_assignment_permanent == 65 and
  .butterfly.independently_derived_first_gate == ["SHEAR",15,16,10] and
  .butterfly.first_gate_orbit_coefficient_mismatch == 60 and
  .controls.wrong_sector_orbit_kernel == 0 and
  (.universal_structured_transform_lower_bound_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
comparisons = {
    "topology": (
        [
            production["quotient_geometry"]["full_exchange_symmetric_occupation_cells"],
            production["quotient_geometry"]["momentum_bracelet_cells"],
            production["quotient_geometry"]["position_zero_total_cells_before_reflection"],
            production["quotient_geometry"]["position_zero_total_reflection_closed_cells"],
        ],
        [
            oracle["topology"]["occupations"],
            oracle["topology"]["momentum_bracelets"],
            oracle["topology"]["zero_total_position_occupations"],
            oracle["topology"]["reflection_closed_position_cells"],
        ],
    ),
    "dense row commitment": (
        production["direct_kernel_route"]["dense_row_commitment"],
        oracle["direct_kernel"]["dense_row_commitment"],
    ),
    "dense column commitment": (
        production["direct_kernel_route"]["dense_column_commitment"],
        oracle["direct_kernel"]["dense_column_commitment"],
    ),
    "generic permanent": (
        production["direct_kernel_route"]["generic_permanent_subset_value"],
        oracle["direct_kernel"]["generic_ryser_permanent"],
    ),
    "first gate": (
        production["standard_butterfly_route"]["first_forward_gate"],
        oracle["butterfly"]["independently_derived_first_gate"],
    ),
}
for label, (left, right) in comparisons.items():
    if left != right:
        raise SystemExit(f"independent dihedral-quotient parity differs: {label}")
PY

"$python" - "$oracle" <<'PY'
import ast, pathlib, sys
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.add(node.module or "")
if any("growing_rotor" in name or "cat_cas" in name for name in imports):
    raise SystemExit("dihedral-quotient oracle imports CAT_CAS production")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_DIHEDRAL_QUOTIENT_BOSONIC_FOURIER_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
