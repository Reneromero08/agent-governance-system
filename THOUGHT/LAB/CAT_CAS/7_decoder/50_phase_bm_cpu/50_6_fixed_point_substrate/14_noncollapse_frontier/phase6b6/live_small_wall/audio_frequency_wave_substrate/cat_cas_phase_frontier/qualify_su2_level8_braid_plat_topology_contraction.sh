#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_su2_level8_braid_plat_topology_contraction.sh MANAGED_BUILD_DIR" >&2
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
production="$here/su2_level8_braid_plat_topology_contraction.py"
reference="$here/su2_level8_braid_plat_topology_contraction_separate_reference.py"
substrate="$here/su2_level8_fusion_path_braid_phase_relation.py"
sealed="$here/SU2_LEVEL8_BRAID_PLAT_TOPOLOGY_CONTRACTION_RESULTS.json"
sealed_reference="$here/SU2_LEVEL8_BRAID_PLAT_TOPOLOGY_CONTRACTION_SEPARATE_REFERENCE.json"

mkdir -p "$build_dir/pycache"
filesystem_type=$(findmnt -n -o FSTYPE -T "$build_dir")
case "$filesystem_type" in
  tmpfs|ramfs)
    echo "managed build directory resolves to a RAM-backed filesystem" >&2
    exit 2
    ;;
esac
export TMPDIR="$build_dir"
export TMP="$build_dir"
export TEMP="$build_dir"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$build_dir/pycache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

nice -n 10 ionice -c 3 "$python" "$production" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("SU2 level-8 plat production differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$reference" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("SU2 level-8 plat reference differs from seal")
' "$sealed_reference"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "SEPARATE_REFERENCE_PARITY" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_EXACT_SU2_LEVEL8_BRAID_PLAT_PUBLIC_TOPOLOGY_CONTRACTION_WITH_DEPTH_WIDTH_OBSTRUCTION" and
  ([.executed_cases[].exact_direct_boundary_agreement] | all) and
  ([.executed_cases[].topology_induced_width] == [3,5,9,11,11,11,11,3,5,9,11,11,11,11]) and
  ([.depth_profile[].topology_induced_width] == [0,1,4,8,11,15,18,21,23]) and
  ([.depth_profile[].peak_support_factor_cells] == [1,2,22,24,245,934,5716,30138,99390]) and
  .phase_relation_law.compiler_reads_exact_coefficients_or_boundary == false and
  .phase_relation_law.intermediate_fusion_rows_projected == false and
  .phase_relation_law.complete_fusion_path_vector_materialized_on_accepted_path == false and
  .phase_relation_law.inverse_rematerializes_contraction_from_public_topology and
  .transaction.primary.restoration_error_field_cells == 0 and
  .transaction.primary.same_accumulator_backing and
  .transaction.primary.canonical_post_restoration_state_exact and
  .transaction.reuse.restoration_error_field_cells == 0 and
  .transaction.reuse.same_accumulator_backing and
  .transaction.reuse.canonical_post_restoration_state_exact and
  .transaction.fresh_restored_reuse_boundary_agreement and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.primary.baseline_reload_used | not) and
  .controls.premature_projection_rejected and
  .controls.wrong_owner_rejected and
  .controls.wrong_public_program_inverse_rejected and
  .controls.missing_inverse_detected and
  .controls.null_carrier_rejected and
  (.controls.snapshot_command_available | not) and
  .resource_law.primary_accumulator_field_cells == 1 and
  .resource_law.primary_peak_live_exact_factor_cells == 771 and
  .resource_law.primary_peak_live_exact_factor_payload_bits == 39127 and
  .resource_law.primary_retained_public_leaf_descriptors == 120 and
  .resource_law.primary_retained_public_leaf_descriptor_integer_cells == 1292 and
  .resource_law.primary_retained_public_leaf_support_assignment_records == 1114 and
  .resource_law.primary_retained_public_leaf_support_label_integer_cells == 4314 and
  .resource_law.primary_peak_compiler_live_support_assignment_records == 1114 and
  .resource_law.primary_peak_compiler_live_support_label_integer_cells == 5416 and
  .resource_law.primary_public_plan_records == 224 and
  .resource_law.primary_public_plan_integer_cells == 1106 and
  .resource_law.primary_direct_verification_fusion_path_cells == 1430 and
  .resource_law.primary_direct_verification_fusion_path_payload_bits == 256269 and
  .resource_law.controller_backend_traffic_bytes == 0 and
  .resource_law.snapshot_traffic_bytes == 0 and
  .resource_law.accepted_retained_inverse_history == 0 and
  .matched_classical_baselines.strongest_compact == "IDENTICAL_EXACT_PUBLIC_TOPOLOGY_SPARSE_FACTOR_ELIMINATION" and
  (.matched_classical_baselines.phase_specific_contraction_reduction | not) and
  (.matched_classical_baselines.computational_advantage | not) and
  .claim_limits.fixed_eight_sweep_strand_growth_boundary_contraction and
  (.claim_limits.fixed_separator_across_growing_sweep_depth | not) and
  (.claim_limits.full_state_compaction | not) and
  (.claim_limits.catvm_custody | not) and
  (.claim_limits.distinct_phase_resource_established | not) and
  (.claim_limits.computational_advantage | not) and
  (.claim_limits.small_wall_crossed | not) and
  (.claim_limits.physical_waveform_execution | not) and
  (.claim_limits.physical_bit_replacement | not) and
  (.claim_limits.catalytic_inference_established | not) and
  (.claim_limits.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "SEPARATE_REFERENCE_PARITY" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.reference_imports_m216_production | not) and
  .reference_algorithm == "INDEPENDENT_FIXED_COLUMN_MAJOR_SPARSE_VARIABLE_ELIMINATION" and
  .underlying_m214_exact_field_and_braid_program_reused and
  .all_cases_exact_direct_boundary_agreement and
  .fixed_eight_sweep_strand_family_scope_only and
  ([.cases[].exact_direct_boundary_agreement] | all) and
  (.distinct_phase_resource_established | not) and
  (.terminal | not)
' "$sealed_reference" >/dev/null

if grep -Eq '(^|[[:space:]])(from|import)[[:space:]]+su2_level8_braid_plat_topology_contraction([[:space:].]|$)' "$reference"; then
  echo "separate reference imports M216 production" >&2
  exit 1
fi

"$python" - "$sealed" "$sealed_reference" "$substrate" <<'PY'
import hashlib, json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
reference = json.load(open(sys.argv[2], encoding="utf-8"))
substrate_hash = hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
if production["source_dependencies"]["m214_production_sha256"] != substrate_hash:
    raise SystemExit("M214 substrate dependency hash mismatch")
production_cases = {
    (case["strands"], case["family"]): case
    for case in production["executed_cases"]
}
reference_cases = {
    (case["strands"], case["family"]): case
    for case in reference["cases"]
}
if production_cases.keys() != reference_cases.keys():
    raise SystemExit("production/reference case domains differ")
for key, candidate in production_cases.items():
    if candidate["boundary_commitment"] != reference_cases[key]["boundary_commitment"]:
        raise SystemExit(f"separate-reference boundary parity failed at {key}")
PY

sha256sum "$production" "$reference" "$substrate" "$sealed" "$sealed_reference"
echo "QUALIFIED_SU2_LEVEL8_BRAID_PLAT_TOPOLOGY_CONTRACTION_STRICT_SCOPE"
echo "evidence=tracked-in-place:$here"
