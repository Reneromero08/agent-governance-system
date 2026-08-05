#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_growing_rotor_refined_boundary_krylov_diagnostic.sh MANAGED_BUILD_DIR" >&2
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
production="$here/growing_rotor_refined_boundary_krylov_diagnostic.py"
oracle="$here/growing_rotor_refined_boundary_krylov_independent_oracle.py"
sealed="$here/GROWING_ROTOR_REFINED_BOUNDARY_KRYLOV_RESULTS.json"
sealed_oracle="$here/GROWING_ROTOR_REFINED_BOUNDARY_KRYLOV_INDEPENDENT_ORACLE.json"

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
    raise SystemExit("refined-boundary Krylov production reexecution differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("refined-boundary Krylov oracle reexecution differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "NO_RESTORATION_CLAIM" and
  .result == "PASS_DIAGNOSTIC_NO_COMPACT_RECURRENCE" and
  .declared_program == {
    "operator_order":"SCATTERING_AFTER_DIAGONAL",
    "prime":103,
    "program_tag":0,
    "rotors":6,
    "seventeenth_root":72,
    "source_family":0,
    "step":0
  } and
  .topology.occupation_histograms == 74613 and
  .topology.necklace_cells == 4389 and
  .topology.bracelet_and_refined_signature_cells == 2277 and
  (.topology.public_topology_compile_reads_final_answers | not) and
  .exact_boundary_recurrence.first_public_word_boundary == 83 and
  .exact_boundary_recurrence.first_public_word_state_commitment == "834956d4d03066d651390a4e2d4b8c0b0940e8169f0b1fb7dfb62d201679c05e" and
  .exact_boundary_recurrence.k0_sequence_degree == 2261 and
  .exact_boundary_recurrence.k0_dimension_deficit == 16 and
  .exact_boundary_recurrence.k0_last_coefficient == 0 and
  .exact_boundary_recurrence.k1_shifted_sequence_degree == 2260 and
  .exact_boundary_recurrence.k1_shifted_dynamic_cell_saving == 17 and
  .exact_boundary_recurrence.k1_shifted_connection_nonzero_coefficients == 2244 and
  .exact_boundary_recurrence.k1_shifted_last_coefficient == 73 and
  .exact_boundary_recurrence.k0_training_violations == 0 and
  .exact_boundary_recurrence.k0_holdout_violations == 0 and
  .exact_boundary_recurrence.k1_training_violations == 0 and
  .exact_boundary_recurrence.k1_holdout_violations == 0 and
  .exact_boundary_recurrence.one_full_state_word_required_before_shifted_recurrence and
  .exact_boundary_recurrence.recurrence_does_not_reconstruct_internal_phase_state and
  (.exact_boundary_recurrence.fixed_rank_or_depth_independent_recurrence_established | not) and
  .controls.reordered_differs and
  .controls.inverse_root_differs and
  .controls.undersampled_out_of_sample_violations > 0 and
  .controls.null_state_rejected and
  .resource_law.accepted_m197_retained_plan_nonzeros == 0 and
  .resource_law.diagnostic_source_shift_plan_entries == 652048 and
  .resource_law.diagnostic_aggregated_csr_nonzeros == 172838 and
  .resource_law.diagnostic_csr_is_verification_only and
  .resource_law.shifted_recurrence_dynamic_field_cells == 2260 and
  .resource_law.shifted_recurrence_public_nonzero_coefficients == 2244 and
  .predecessor_restoration_evidence.restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .predecessor_restoration_evidence.this_diagnostic_does_not_reexecute_or_extend_that_claim and
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
  .restoration_classification == "NO_RESTORATION_CLAIM" and
  .result == "PASS_DIAGNOSTIC_NO_COMPACT_RECURRENCE" and
  .topology.occupation_histograms == 74613 and
  .topology.necklace_cells == 4389 and
  .topology.bracelet_and_refined_signature_cells == 2277 and
  .operator.raw_transition_terms == 684624 and
  .operator.raw_transition_commitment == "1774f44283dde49154d3aec11bb0dd22179a061290e44448c795402fa2ca0465" and
  .operator.aggregated_csr_nonzeros == 172838 and
  .exact_boundary_recurrence.k0_degree == 2261 and
  .exact_boundary_recurrence.k1_degree == 2260 and
  .exact_boundary_recurrence.k1_dynamic_cell_saving == 17 and
  .exact_boundary_recurrence.k1_nonzero_coefficients == 2244 and
  .exact_boundary_recurrence.k0_training_violations == 0 and
  .exact_boundary_recurrence.k0_holdout_violations == 0 and
  .exact_boundary_recurrence.k1_training_violations == 0 and
  .exact_boundary_recurrence.k1_holdout_violations == 0 and
  (.production_source_imported | not) and
  (.production_projection_called | not) and
  (.production_bm_called | not) and
  .oracle_sparse_operator_is_verification_only and
  (.fixed_rank_or_depth_independent_recurrence_established | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json
import sys

production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
p_topology = production["topology"]
o_topology = oracle["topology"]
for p_key, o_key in (
    ("occupation_histograms", "occupation_histograms"),
    ("necklace_cells", "necklace_cells"),
    ("bracelet_and_refined_signature_cells", "bracelet_and_refined_signature_cells"),
    ("topology_commitment", "topology_commitment"),
):
    if p_topology[p_key] != o_topology[o_key]:
        raise SystemExit(f"independent topology metric differs: {p_key}")
p_recurrence = production["exact_boundary_recurrence"]
o_recurrence = oracle["exact_boundary_recurrence"]
for p_key, o_key in (
    ("sequence_commitment", "sequence_commitment"),
    ("sequence_prefix", "sequence_prefix"),
    ("first_public_word_boundary", "first_public_word_boundary"),
    ("first_public_word_state_commitment", "first_public_word_state_commitment"),
    ("k0_sequence_degree", "k0_degree"),
    ("k0_connection_slots", "k0_connection_slots"),
    ("k0_connection_nonzero_coefficients", "k0_nonzero_coefficients"),
    ("k0_last_coefficient", "k0_last_coefficient"),
    ("k0_connection_commitment", "k0_connection_commitment"),
    ("k1_shifted_sequence_degree", "k1_degree"),
    ("k1_shifted_dynamic_cell_saving", "k1_dynamic_cell_saving"),
    ("k1_shifted_connection_slots", "k1_connection_slots"),
    ("k1_shifted_connection_nonzero_coefficients", "k1_nonzero_coefficients"),
    ("k1_shifted_last_coefficient", "k1_last_coefficient"),
    ("k1_connection_commitment", "k1_connection_commitment"),
):
    if p_recurrence[p_key] != o_recurrence[o_key]:
        raise SystemExit(f"independent recurrence metric differs: {p_key}")
if production["resource_law"]["diagnostic_csr_commitment"] != oracle["operator"]["aggregated_csr_commitment"]:
    raise SystemExit("independent CSR commitment differs")
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
    if any("growing_rotor" in name or "cat_cas" in name for name in names):
        raise SystemExit("refined-boundary oracle production dependency isolation failed")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_GROWING_ROTOR_REFINED_BOUNDARY_KRYLOV_DIAGNOSTIC_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
