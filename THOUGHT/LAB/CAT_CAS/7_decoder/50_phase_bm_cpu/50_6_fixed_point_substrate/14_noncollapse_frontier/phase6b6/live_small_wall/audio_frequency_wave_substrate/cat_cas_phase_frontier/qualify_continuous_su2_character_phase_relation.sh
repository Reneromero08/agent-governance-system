#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_continuous_su2_character_phase_relation.sh MANAGED_BUILD_DIR" >&2
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
production="$here/continuous_su2_character_phase_relation_closure.py"
oracle="$here/continuous_su2_character_phase_relation_independent_oracle.py"
sealed="$here/CONTINUOUS_SU2_CHARACTER_PHASE_RELATION_RESULTS.json"
sealed_oracle="$here/CONTINUOUS_SU2_CHARACTER_PHASE_RELATION_INDEPENDENT_ORACLE.json"

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
    raise SystemExit("continuous SU2 production differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("continuous SU2 oracle differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_EXACT_CONTINUOUS_SU2_CHARACTER_RELATION_CLOSURE_WITH_GROWING_REPRESENTATION_SUPPORT" and
  .phase_relation_law.domain == "CONTINUOUS_SU2_NO_FINITE_GROUP_ENUMERATION" and
  .phase_relation_law.relation_composition == "NORMALIZED_HAAR_CONVOLUTION_DIAGONAL_CASIMIR_SPECTRAL_PHASE" and
  .phase_relation_law.relation_intersection == "POINTWISE_CLEBSCH_GORDAN_FUNDAMENTAL_FUSION" and
  .phase_relation_law.shared_unresolved_group_ports == 1 and
  .phase_relation_law.multiple_noncommuting_consumers and
  (.phase_relation_law.intermediate_character_projection | not) and
  (.phase_relation_law.truth_table_assignment_or_group_element_expansion | not) and
  (.phase_relation_law.finite_group_reduction | not) and
  .analytic_support_law.highest_weight_at_depth_d == "D_PLUS_1" and
  .analytic_support_law.character_cells_at_depth_d == "D_PLUS_2" and
  (.analytic_support_law.fixed_character_support_for_declared_growing_depth_family | not) and
  .transaction.primary_boundary_commitment == "f003318cea8af1e9f7d5005c11682e94acf19bed9956ff03e8a57d695b6ce8c2" and
  .transaction.primary_boundary_payload_bits == 156592 and
  .transaction.reuse_boundary_commitment == "6f96b47dc010741c851b103fcc88a597a274a8ba18f9ca765f0d60db7665d1db" and
  .transaction.reuse_boundary_commitment == .transaction.fresh_reuse_boundary_commitment and
  .transaction.fresh_restored_reuse_boundary_agreement and
  .transaction.fresh_restored_reuse_state_agreement and
  .transaction.primary_restoration_error_character_cells == 0 and
  .transaction.reuse_restoration_error_character_cells == 0 and
  .transaction.primary_same_backing and .transaction.reuse_same_backing and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .controls.wrong_owner_rejected and
  .controls.wrong_operation_type_rejected and
  .controls.premature_projection_rejected and
  .controls.missing_inverse_detected and
  .controls.reordered_inverse_rejected and
  .controls.null_carrier_rejected and
  .controls.module_order_noncommuting_mismatch_cells > 0 and
  .controls.semantic_fusion_perturbation_changes_boundary and
  .resource_law.initial_character_cells == 2 and
  .resource_law.compiled_public_parameter_gaussian_rational_cells == 5 and
  .resource_law.primary_forward_character_cells == 34 and
  .resource_law.primary_forward_character_payload_bits == 3728007 and
  .resource_law.primary_boundary_payload_bits == 156592 and
  .resource_law.primary_full_transaction_work.public_program_commitment_hashes == 131 and
  .resource_law.primary_full_transaction_work.public_program_operation_records_hashed == 8384 and
  .resource_law.primary_full_transaction_work.state_commitment_hashes == 1 and
  .resource_law.primary_full_transaction_work.state_commitment_character_cells_hashed == 34 and
  .resource_law.primary_full_transaction_work.boundary_commitment_hashes == 1 and
  .resource_law.retained_inverse_history_entries == 0 and
  .resource_law.additional_retained_plan_entries == 0 and
  .resource_law.temporary_full_character_vector_cells == 0 and
  .resource_law.controller_backend_traffic_not_applicable_direct_process and
  .matched_classical_baselines.strongest_compact == "IDENTICAL_EXACT_GAUSSIAN_RATIONAL_SU2_CHARACTER_COEFFICIENT_RECURRENCE" and
  (.matched_classical_baselines.strictly_smaller_or_faster_phase_path_established | not) and
  (.catvm_custody | not) and
  (.distinct_phase_resource_established | not) and
  (.computational_advantage | not) and
  (.small_wall_crossed | not) and
  (.physical_waveform_execution | not) and
  (.physical_bit_replacement | not) and
  (.catalytic_inference_established | not) and
  (.unbounded_computation_established | not) and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.oracle_imports_cat_cas_modules | not) and
  .primary.character_cells == 34 and
  .primary.character_payload_bits == 3728007 and
  .primary.polynomial_cells == 34 and
  .primary.polynomial_payload_bits == 5323677 and
  .primary.boundary_payload_bits == 156592 and
  .primary_resource_tuple_reproduced and
  .module_order_polynomials_differ and
  .module_order_boundaries_differ and
  (.continuous_su2_group_element_enumeration_used | not) and
  (.finite_group_reduction_used | not) and
  (.distinct_phase_resource_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
production_cases = {
    (case["depth"], case["family"]): case
    for case in production["depth_cases"]
}
for case in oracle["cases"]:
    other = production_cases[(case["depth"], case["family"])]
    pairs = (
        ("character_cells", "character_coefficient_cells"),
        ("character_payload_bits", "character_payload_bits"),
        ("state_commitment", "state_commitment"),
        ("boundary_commitment", "boundary_commitment"),
        ("boundary_payload_bits", "boundary_payload_bits"),
    )
    for oracle_field, production_field in pairs:
        if case[oracle_field] != other[production_field]:
            raise SystemExit(
                f"continuous SU2 case parity differs: {case['depth']=} "
                f"{case['family']=} {oracle_field=}"
            )
if production["transaction"]["primary_boundary_commitment"] != oracle["primary"]["boundary_commitment"]:
    raise SystemExit("continuous SU2 primary boundary parity differs")
if production["transaction"]["reuse_boundary_commitment"] != oracle["reuse"]["boundary_commitment"]:
    raise SystemExit("continuous SU2 reuse boundary parity differs")
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
if any("continuous_su2" in name or "cat_cas" in name for name in imports):
    raise SystemExit("continuous SU2 oracle imports CAT_CAS production")
PY

"$python" -m py_compile "$production" "$oracle"
sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_CONTINUOUS_SU2_CHARACTER_PHASE_RELATION_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
