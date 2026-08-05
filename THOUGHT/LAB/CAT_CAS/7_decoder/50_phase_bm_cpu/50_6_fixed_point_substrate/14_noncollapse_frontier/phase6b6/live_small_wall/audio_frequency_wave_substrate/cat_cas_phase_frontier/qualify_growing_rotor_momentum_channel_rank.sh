#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_momentum_channel_rank.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_momentum_channel_rank_diagnostic.py"
oracle="$here/growing_rotor_momentum_channel_rank_independent_oracle.py"
sealed="$here/GROWING_ROTOR_MOMENTUM_CHANNEL_RANK_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_MOMENTUM_CHANNEL_RANK_INDEPENDENT_ORACLE.json"

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
    raise SystemExit("momentum-channel production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("momentum-channel oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NO_RESTORATION_CLAIM" and
  .result == "PASS_RANK8_EXACT_F103_PORT_CELL_AND_UNIFORM_LINEAR_OPERATOR_QUOTIENT_BELOW8_REJECTED" and
  .certificate.channels == [1,2,3,4,5,6,7,8] and
  .certificate.port_reachability_rank == 8 and
  .certificate.port_reachability_determinant == 98 and
  .certificate.port_image_cardinality == "12667700813876161" and
  .certificate.exact_f103_port_cell_lower_bound == 8 and
  .certificate.operator_witness_rank == 8 and
  .certificate.operator_witness_determinant == 50 and
  .certificate.public_weight_witness_rank == 8 and
  .certificate.public_weight_witness_determinant == 80 and
  .certificate.linear_quotient_lower_bound == 8 and
  .controls.drop_port_row_rank == 7 and
  .controls.drop_operator_row_rank == 7 and
  .controls.drop_public_weight_row_rank == 7 and
  .controls.duplicate_port_column_rank == 7 and
  .controls.duplicate_operator_column_rank == 7 and
  .controls.duplicate_public_weight_row_rank == 7 and
  .controls.source_target_coordinates_distinct and
  .resource_law.port_witness_coordinate_comparisons == 64 and
  .resource_law.operator_candidate_terms_streamed == 896 and
  .resource_law.operator_witness_contributions_accepted == 16 and
  .resource_law.public_weight_candidates_streamed == 8 and
  .resource_law.peak_named_field_and_descriptor_slots == 292 and
  .resource_law.dense_2277_squared_operator_cells == 0 and
  .resource_law.full_74613_occupation_scratch_cells == 0 and
  .resource_law.retained_transition_plan_entries == 0 and
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
  .result == "PASS_RANK8_EXACT_F103_PORT_CELL_AND_UNIFORM_LINEAR_OPERATOR_QUOTIENT_BELOW8_REJECTED" and
  .independent_topology.occupations_enumerated == 74613 and
  .independent_topology.necklaces_enumerated == 4389 and
  .independent_topology.bracelets_enumerated == 2277 and
  .independent_topology.all_witness_coordinates_present and
  .independent_topology.full_topology_is_verification_only and
  .independent_certificate.port_reachability_rank == 8 and
  .independent_certificate.port_reachability_permutation_determinant == 98 and
  .independent_certificate.operator_rank == 8 and
  .independent_certificate.operator_permutation_determinant == 50 and
  .independent_certificate.direct_factor_witness_agreement and
  .independent_certificate.public_weight_rank == 8 and
  .independent_certificate.all_seven_families_seventeen_steps_rank == 8 and
  .independent_certificate.public_weight_permutation_determinant == 80 and
  .controls.drop_port_row_rank == 7 and
  .controls.drop_operator_row_rank == 7 and
  .controls.drop_public_weight_row_rank == 7 and
  .controls.duplicate_port_column_rank == 7 and
  .controls.duplicate_operator_column_rank == 7 and
  .controls.duplicate_public_weight_row_rank == 7 and
  .resource_derivation.accepted_peak_named_field_and_descriptor_slots == 292 and
  .resource_derivation.accepted_dense_operator_cells == 0 and
  .resource_derivation.accepted_occupation_scratch_cells == 0 and
  .resource_derivation.accepted_transition_plan_entries == 0 and
  (.production_module_imported | not) and
  (.nonlinear_or_program_specialized_quotients_rejected | not) and
  (.catvm_custody | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.physical_waveform_execution | not) and
  (.physical_bit_replacement | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
comparisons = {
    "port witness": (production["certificate"]["port_reachability_witness"], oracle["independent_certificate"]["port_reachability_witness"]),
    "port commitment": (production["certificate"]["port_reachability_witness_commitment"], oracle["independent_certificate"]["port_reachability_witness_commitment"]),
    "operator witness": (production["certificate"]["operator_witness"], oracle["independent_certificate"]["direct_operator_witness"]),
    "operator commitment": (production["certificate"]["operator_witness_commitment"], oracle["independent_certificate"]["operator_witness_commitment"]),
    "weight witness": (production["certificate"]["public_weight_witness"], oracle["independent_certificate"]["public_weight_witness"]),
    "weight commitment": (production["certificate"]["public_weight_witness_commitment"], oracle["independent_certificate"]["public_weight_witness_commitment"]),
    "descriptors": (production["certificate"]["public_program_descriptors"], oracle["independent_certificate"]["public_program_descriptors"]),
    "source code": (production["certificate"]["operator_coordinate_source_bracelet_code"], oracle["independent_topology"]["source_bracelet_code"]),
    "target codes": (production["certificate"]["operator_coordinate_target_bracelet_codes"], oracle["independent_topology"]["target_bracelet_codes"]),
}
for label, (left, right) in comparisons.items():
    if left != right:
        raise SystemExit(f"independent momentum-channel parity differs: {label}")
PY

"$python" - "$production" "$oracle" <<'PY'
import ast, pathlib, sys

def imported(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(node.module or "")
    return result

production_imports = imported(sys.argv[1])
allowed = {"growing_rotor_coherent_momentum_wave_streaming_closure"}
forbidden = {name for name in production_imports if ("growing_rotor" in name or "cat_cas" in name) and name not in allowed}
if forbidden:
    raise SystemExit(f"momentum-channel production dependency changed: {sorted(forbidden)}")

if any("growing_rotor" in name or "cat_cas" in name for name in imported(sys.argv[2])):
    raise SystemExit("momentum-channel oracle imports CAT_CAS production")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_MOMENTUM_CHANNEL_RANK_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
