#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_su2_level8_fusion_path_braid.sh MANAGED_BUILD_DIR" >&2
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
production="$here/su2_level8_fusion_path_braid_phase_relation.py"
oracle="$here/su2_level8_fusion_path_braid_independent_oracle.py"
sealed="$here/SU2_LEVEL8_FUSION_PATH_BRAID_PHASE_RELATION_RESULTS.json"
sealed_oracle="$here/SU2_LEVEL8_FUSION_PATH_BRAID_INDEPENDENT_ORACLE.json"

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
    raise SystemExit("SU2 level-8 fusion-path production differs from seal")
' "$sealed"

nice -n 10 ionice -c 3 "$python" "$oracle" | "$python" -c '
import json, sys
sealed = json.load(open(sys.argv[1], encoding="utf-8"))
fresh = json.load(sys.stdin)
if sealed != fresh:
    raise SystemExit("SU2 level-8 fusion-path oracle differs from seal")
' "$sealed_oracle"

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS_EXACT_SU2_LEVEL8_FUSION_PATH_BRAID_CLOSURE_WITH_GROWING_GLOBAL_PATH_WIDTH" and
  .phase_relation_law.domain == "SU2_LEVEL8_TEMPERLEY_LIEB_A9_VACUUM_FUSION_PATH_SPACE" and
  .phase_relation_law.coefficient_field == "Q_ZETA40_DEGREE16" and
  .phase_relation_law.local_simple_object_labels == 9 and
  .phase_relation_law.radical_free_nonorthogonal_path_gauge and
  .phase_relation_law.multiple_noncommuting_consumers and
  .phase_relation_law.shared_internal_channels_unprojected and
  (.phase_relation_law.intermediate_projection | not) and
  (.phase_relation_law.retained_fusion_path_list | not) and
  (.phase_relation_law.retained_local_action_plan | not) and
  .phase_relation_law.public_path_count_topology_does_not_inspect_boundary and
  .phase_relation_law.global_fusion_path_coefficient_vector_materialized and
  [.executed_cases[].fusion_path_field_cells] == [2,5,14,42,132,429,1430] and
  ([.executed_cases[].nonzero_fusion_path_field_cells] == [2,5,14,42,132,429,1430]) and
  .dimension_law[8].strands == 18 and
  .dimension_law[8].su2_level8_vacuum_path_cells == 4861 and
  .dimension_law[8].untruncated_catalan_cells == 4862 and
  .dimension_law[8].jones_wenzl_removed_cells == 1 and
  .transaction.primary_boundary_commitment == "ff38debbe990ce551d2292914c8be41afec921757e521742fdbd5b6bdca63447" and
  .transaction.primary_forward_state_commitment == "6f228996bd6885bbf4b9bc1719e5168d48cb1ba2a8d686bd38d607e8d7b4127a" and
  .transaction.primary_forward_field_cells == 1430 and
  .transaction.primary_forward_payload_bits == 256269 and
  .transaction.primary_restoration_error_field_cells == 0 and
  .transaction.primary_same_coefficient_backing and
  .transaction.primary_canonical_post_restoration_state_exact and
  .transaction.reuse_boundary_commitment == "4ca9e758933ad3f984878b7a96e9ab9991b92a8c3f9e65975974c3862ef15634" and
  .transaction.reuse_forward_state_commitment == "66bdc666c1c5b0436544656787cd2214aa573da916f9740a2ee5e9c84b80c411" and
  .transaction.reuse_restoration_error_field_cells == 0 and
  .transaction.reuse_same_coefficient_backing and
  .transaction.reuse_canonical_post_restoration_state_exact and
  .transaction.fresh_restored_reuse_boundary_agreement and
  .transaction.fresh_restored_reuse_state_agreement and
  .transaction.restoration_generation_after_reuse == 2 and
  (.transaction.primary_baseline_reload_used | not) and
  (.transaction.reuse_baseline_reload_used | not) and
  .controls.yang_baxter_relation_exact and
  .controls.far_generators_commute_exactly and
  .controls.adjacent_generator_noncommutation_mismatch_cells > 0 and
  .controls.single_generator_inverse_exact and
  .controls.reordered_inverse_changes_state_cells > 0 and
  .controls.semantic_generator_perturbation_changes_state_cells > 0 and
  .controls.wrong_owner_rejected and
  .controls.wrong_operation_type_rejected and
  .controls.premature_projection_rejected and
  .controls.missing_inverse_detected and
  .controls.reordered_inverse_rejected and
  .controls.null_carrier_rejected and
  (.controls.snapshot_command_available | not) and
  .resource_law.category_constant_field_cells == 20 and
  .resource_law.primary_public_topology_integer_cells == 153 and
  .resource_law.primary_public_topology_payload_bits == 516 and
  .resource_law.primary_public_program_descriptor_integers == 3 and
  .resource_law.primary_public_operation_records_materialized == 0 and
  .resource_law.primary_retained_path_records == 0 and
  .resource_law.primary_retained_local_action_records == 0 and
  .resource_law.primary_streamed_path_label_scratch_cells == 34 and
  .resource_law.primary_local_field_scratch_cells == 8 and
  .resource_law.primary_forward_carrier_field_cells == 1430 and
  .resource_law.primary_forward_carrier_payload_bits == 256269 and
  .resource_law.primary_full_transaction_work.forward_operations == 120 and
  .resource_law.primary_full_transaction_work.inverse_operations == 120 and
  .resource_law.primary_full_transaction_work.path_unrank_calls == 343200 and
  .resource_law.primary_full_transaction_work.path_rank_calls == 80081 and
  .resource_law.primary_full_transaction_work.path_labels_rematerialized == 7195777 and
  .resource_law.primary_full_transaction_work.topology_count_reads == 7877584 and
  .resource_law.retained_inverse_history_entries == 0 and
  .resource_law.additional_retained_plan_entries == 0 and
  .matched_classical_baselines.strongest_compact == "IDENTICAL_EXACT_QZETA40_TEMPERLEY_LIEB_FUSION_PATH_RECURRENCE_WITH_PUBLIC_A9_RANK_UNRANK" and
  (.matched_classical_baselines.strictly_smaller_or_faster_phase_path_established | not) and
  .claim_limits.fixed_local_label_alphabet and
  (.claim_limits.fixed_global_carrier_across_growing_strands | not) and
  (.claim_limits.bounded_exact_coefficient_width | not) and
  .claim_limits.fusion_path_enumeration_materialized_as_coefficients and
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
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  (.oracle_imports_cat_cas_modules | not) and
  .production_uses_source_weight_gauge and
  .oracle_uses_distinct_target_weight_gauge and
  .oracle_explicit_path_list_and_index_map_is_a_VERIFICATION_BASELINE and
  [.cases[].fusion_path_field_cells] == [2,5,14,42,132,429,1430] and
  .primary.source_restored and
  .primary.same_backing and
  (.primary.baseline_reload_used | not) and
  .reuse.source_restored and
  .reuse.same_backing and
  (.reuse.baseline_reload_used | not) and
  .controls.yang_baxter_relation_exact and
  .controls.far_generators_commute_exactly and
  .controls.adjacent_generators_do_not_commute and
  .controls.single_generator_inverse_exact and
  .jones_wenzl_relation_exact and
  .first_truncation_at_strands18_reproduced and
  (.distinct_phase_resource_established | not) and
  (.terminal | not)
' "$sealed_oracle" >/dev/null

"$python" - "$sealed" "$sealed_oracle" <<'PY'
import json, sys
production = json.load(open(sys.argv[1], encoding="utf-8"))
oracle = json.load(open(sys.argv[2], encoding="utf-8"))
keys = (
    "strands", "rounds", "family", "program_steps",
    "fusion_path_field_cells", "nonzero_fusion_path_field_cells",
    "carrier_payload_bits", "maximum_signed_numerator_bits",
    "maximum_denominator_bits", "state_commitment", "boundary_commitment",
    "boundary_payload_bits", "public_topology_integer_cells",
    "public_topology_payload_bits", "topology_commitment",
)
left = [{key: case[key] for key in keys} for case in production["executed_cases"]]
right = [{key: case[key] for key in keys} for case in oracle["cases"]]
if left != right:
    raise SystemExit("independent fusion-path case parity failed")
if production["dimension_law"] != oracle["dimension_law"]:
    raise SystemExit("independent fusion-path dimension law failed")
transaction = production["transaction"]
if transaction["primary_boundary_commitment"] != oracle["primary"]["boundary_commitment"]:
    raise SystemExit("independent primary boundary parity failed")
if transaction["primary_forward_state_commitment"] != oracle["primary"]["forward_state_commitment"]:
    raise SystemExit("independent primary state parity failed")
if transaction["reuse_boundary_commitment"] != oracle["reuse"]["boundary_commitment"]:
    raise SystemExit("independent reuse boundary parity failed")
if transaction["reuse_forward_state_commitment"] != oracle["reuse"]["forward_state_commitment"]:
    raise SystemExit("independent reuse state parity failed")
PY

sha256sum "$production" "$oracle" "$sealed" "$sealed_oracle"
echo "QUALIFIED_SU2_LEVEL8_FUSION_PATH_BRAID_STRICT_SCOPE"
echo "evidence=tracked-in-place:$here"
