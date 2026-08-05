#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_pair_signature_dihedral_transfer_no_go.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_pair_signature_dihedral_transfer_no_go.py"
oracle="$here/growing_rotor_pair_signature_dihedral_transfer_no_go_independent_oracle.py"
sealed="$here/GROWING_ROTOR_PAIR_SIGNATURE_DIHEDRAL_TRANSFER_NO_GO_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_PAIR_SIGNATURE_DIHEDRAL_TRANSFER_NO_GO_INDEPENDENT_ORACLE.json"

mkdir -p "$build_dir" "$build_dir/pycache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$build_dir/pycache"

nice -n 10 ionice -c 3 "$python" "$production" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("dihedral transfer production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("dihedral transfer oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NO_RESTORATION_CLAIM" and
  .result == "PASS_TRANSFER_NO_GO" and
  .burnside_law == "BRACELETS_EQUALS_HALF_OF_NECKLACES_PLUS_REFLECTION_FIXED_NECKLACES_FOR_ODD_GRID17_AND_ROTORS_BELOW17" and
  ([.cases[].rotors] == [2,3,4,5,6]) and
  ([.cases[].occupation_histograms] == [153,969,4845,20349,74613]) and
  ([.cases[].necklace_cells] == [9,57,285,1197,4389]) and
  ([.cases[].burnside_bracelet_cells] == [9,33,165,621,2277]) and
  ([.cases[].pair_signature_cells] == [9,33,165,621,2253]) and
  ([.cases[0:4][].pair_signatures_equal_dihedral_orbits] | all) and
  ([.cases[0:4][].all_sixteen_shift_bases_equitable] | all) and
  ([.cases[0:4][].non_dihedral_pair_signature_fibers] | max == 0) and
  .cases[4].non_dihedral_pair_signature_fibers == 24 and
  .cases[4].nonequitable_pair_signature_fibers == 24 and
  (.cases[4].all_sixteen_shift_bases_equitable | not) and
  .cases[4].first_nonequitable_witness.signed_shift == 1 and
  .cases[4].first_nonequitable_witness.differing_destination_count == 39 and
  .cases[4].first_nonequitable_witness.left_row_sum == 30 and
  .cases[4].first_nonequitable_witness.right_row_sum == 30 and
  (.cases[4].first_nonequitable_witness.targets_are_dihedrally_related | not) and
  (.cases[4].first_nonequitable_witness | has("differences") | not) and
  (.pair_signature_only_analytic_kernel_transferable_through_rotor6 | not) and
  .higher_order_signature_repair_required and
  .matched_classical_diagnostic == "IDENTICAL_HOMOMETRY_AND_SIGNED_SHIFT_PROFILE_COMPARISON" and
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
  .result == "PASS_TRANSFER_NO_GO" and
  (.production_source_imported | not) and
  (.production_transition_called | not) and
  .unordered_particle_signature_used and
  .ordered_particle_shift_used and
  ([.cases[].rotors] == [2,3,4,5,6])
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
if production["cases"] != oracle["cases"]:
    raise SystemExit("independent dihedral-transfer case records differ")
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
    if any("growing_rotor_pair_signature_dihedral_transfer_no_go" in name for name in names):
        raise SystemExit("dihedral-transfer oracle production dependency isolation failed")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_PAIR_SIGNATURE_DIHEDRAL_TRANSFER_NO_GO_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
