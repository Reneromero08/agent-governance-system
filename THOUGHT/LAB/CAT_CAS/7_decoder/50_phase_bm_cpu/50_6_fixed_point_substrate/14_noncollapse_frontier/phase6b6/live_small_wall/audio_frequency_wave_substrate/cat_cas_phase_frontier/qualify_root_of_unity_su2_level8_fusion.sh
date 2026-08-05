#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_root_of_unity_su2_level8_fusion.sh MANAGED_BUILD_DIR" >&2
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
production="$here/root_of_unity_su2_level8_fusion_phase_relation.py"
oracle="$here/root_of_unity_su2_level8_fusion_independent_oracle.py"
sealed="$here/ROOT_OF_UNITY_SU2_LEVEL8_FUSION_PHASE_RELATION_RESULTS.json"
sealed_oracle="$here/ROOT_OF_UNITY_SU2_LEVEL8_FUSION_INDEPENDENT_ORACLE.json"

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
    raise SystemExit("SU2 level-8 production differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("SU2 level-8 oracle differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_EXACT_SU2_LEVEL8_FIXED9_FUSION_CLOSURE_WITH_GROWING_COEFFICIENT_HEIGHT" and
  .phase_relation_law.domain == "ROOT_OF_UNITY_SU2_LEVEL8_VERLINDE_FUSION_ALGEBRA" and
  .phase_relation_law.coefficient_field == "Q_ZETA40_DEGREE16" and
  .phase_relation_law.simple_object_field_cells == 9 and
  .phase_relation_law.jones_wenzl_relation == "X9_EQUALS_ZERO" and
  .phase_relation_law.multiple_noncommuting_consumers and
  (.phase_relation_law.intermediate_projection | not) and
  (.phase_relation_law.truth_table_assignment_group_element_or_path_expansion | not) and
  .support_law.fixed_across_declared_depths and
  .support_law.root_of_unity_quotient_is_a_changed_algebra_not_a_lossless_encoding_of_general_continuous_su2 and
  (.support_law.bounded_width_exact_state_set_established | not) and
  .transaction.primary_boundary_commitment == "f5051d1280076a8fd463b72fb86fd69aa1907311f2a33e0d8ffe39c6895f6d5b" and
  .transaction.primary_boundary_payload_bits == 1761 and
  .transaction.reuse_boundary_commitment == "b1224f22ab962d127eb26010385eac099d4d2e024b82c744a5c71f0a172ff957" and
  .transaction.reuse_boundary_commitment == .transaction.fresh_reuse_boundary_commitment and
  .transaction.fresh_restored_reuse_boundary_agreement and
  .transaction.fresh_restored_reuse_state_agreement and
  .transaction.primary_restoration_error_field_cells == 0 and
  .transaction.reuse_restoration_error_field_cells == 0 and
  .transaction.primary_same_coefficient_backing and
  .transaction.primary_same_scratch_backing and
  .transaction.reuse_same_coefficient_backing and
  .transaction.reuse_same_scratch_backing and
  .transaction.canonical_post_restoration_state_exact and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.baseline_reload_used | not) and
  .controls.wrong_owner_rejected and
  .controls.wrong_operation_type_rejected and
  .controls.premature_projection_rejected and
  .controls.missing_inverse_detected and
  .controls.reordered_inverse_rejected and
  .controls.null_carrier_rejected and
  .controls.wrong_inverse_factor_changes_state and
  .controls.module_order_noncommuting_mismatch_cells > 0 and
  .controls.semantic_fusion_perturbation_changes_boundary and
  (.controls.snapshot_command_available | not) and
  .resource_law.initial_carrier_field_cells == 9 and
  .resource_law.initial_carrier_payload_bits == 290 and
  .resource_law.inverse_scratch_field_cells == 9 and
  .resource_law.primary_forward_carrier_field_cells == 9 and
  .resource_law.primary_forward_carrier_payload_bits == 15330 and
  .resource_law.primary_public_operation_records_materialized == 0 and
  .resource_law.primary_public_descriptor_integers == 2 and
  .resource_law.primary_generated_operation_visits == 512 and
  .resource_law.primary_full_transaction_work.peak_carrier_payload_bits == 18083 and
  .resource_law.primary_full_transaction_work.peak_inverse_scratch_payload_bits == 1593 and
  .resource_law.primary_full_transaction_work.peak_carrier_plus_scratch_payload_bits == 19674 and
  .resource_law.primary_full_transaction_work.inverse_pivot_cells_materialized == 1152 and
  .resource_law.primary_full_transaction_work.inverse_pivot_cells_uncomputed == 1152 and
  .resource_law.primary_full_transaction_work.field_inversions == 4224 and
  .resource_law.primary_full_transaction_work.public_program_descriptor_hashes == 515 and
  .resource_law.primary_full_transaction_work.public_program_descriptor_integers_hashed == 1030 and
  .resource_law.retained_inverse_pivot_plan_cells == 0 and
  .resource_law.retained_inverse_history_entries == 0 and
  .resource_law.additional_retained_plan_entries == 0 and
  .resource_law.controller_backend_traffic_not_applicable_direct_process and
  .matched_classical_baselines.strongest_compact == "IDENTICAL_NINE_QZETA40_COORDINATE_SU2_LEVEL8_FUSION_AND_TWIST_RECURRENCE" and
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
  .fusion_inverse_polynomials_verified and
  .jones_wenzl_polynomial_vanishes_at_quantum_dimension and
  .module_order_polynomials_differ and
  .module_order_boundaries_differ and
  .primary.simple_object_field_cells == 9 and
  .primary.simple_object_payload_bits == 15330 and
  .primary.ordinary_polynomial_cells == 9 and
  .primary.ordinary_polynomial_payload_bits == 15660 and
  .primary.boundary_payload_bits == 1761 and
  .primary.source_restored and
  .reuse.source_restored and
  .primary_resource_tuple_reproduced and
  (.path_or_group_element_enumeration_used | not) and
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
        ("simple_object_field_cells", "simple_object_field_cells"),
        ("nonzero_simple_object_field_cells", "nonzero_simple_object_field_cells"),
        ("simple_object_payload_bits", "carrier_payload_bits"),
        ("state_commitment", "state_commitment"),
        ("boundary_commitment", "boundary_commitment"),
        ("boundary_payload_bits", "boundary_payload_bits"),
    )
    for oracle_field, production_field in pairs:
        if case[oracle_field] != other[production_field]:
            raise SystemExit(
                f"SU2 level-8 case parity differs: {case['depth']=} "
                f"{case['family']=} {oracle_field=}"
            )
if production["transaction"]["primary_boundary_commitment"] != oracle["primary"]["boundary_commitment"]:
    raise SystemExit("SU2 level-8 primary boundary parity differs")
if production["transaction"]["reuse_boundary_commitment"] != oracle["reuse"]["boundary_commitment"]:
    raise SystemExit("SU2 level-8 reuse boundary parity differs")
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
if any("root_of_unity_su2" in name or "cat_cas" in name for name in imports):
    raise SystemExit("SU2 level-8 oracle imports CAT_CAS production")
PY

"$python" -m py_compile "$production" "$oracle"
sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_ROOT_OF_UNITY_SU2_LEVEL8_FUSION_STRICT_SCOPE'
printf 'evidence=tracked-in-place:%s\n' "$here"
