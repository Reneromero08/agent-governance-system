#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
  echo "usage: qualify_catvm_growing_rotor_open_momentum.sh MANAGED_BUILD_DIR" >&2
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
controller="$here/catvm_growing_rotor_open_momentum_controller.py"
service="$here/catvm_growing_rotor_open_momentum_service.py"
oracle="$here/catvm_growing_rotor_open_momentum_independent_oracle.py"
sealed="$here/CATVM_GROWING_ROTOR_OPEN_MOMENTUM_RESULTS.json"
sealed_oracle="$here/CATVM_GROWING_ROTOR_OPEN_MOMENTUM_INDEPENDENT_ORACLE.json"

mkdir -p "$build_dir/pycache" "$build_dir/controller-audit" "$build_dir/oracle-audit"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$build_dir/pycache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

files=(
  catvm_growing_rotor_open_momentum_protocol.py
  catvm_growing_rotor_open_momentum_service.py
  catvm_growing_rotor_open_momentum_controller.py
  catvm_growing_rotor_open_momentum_independent_oracle.py
  CATVM_GROWING_ROTOR_OPEN_MOMENTUM_RESULTS.json
  CATVM_GROWING_ROTOR_OPEN_MOMENTUM_INDEPENDENT_ORACLE.json
  CATVM_GROWING_ROTOR_OPEN_MOMENTUM_INDEPENDENT_REEXECUTION_REVIEW.md
  growing_rotor_open_momentum_factor_closure.py
  growing_rotor_open_momentum_factor_independent_oracle.py
  growing_rotor_pair_signature_streamed_quotient.py
)
hashes=(
  06f03fef5ab935536cf9036988ff79a4db69e1b7533215a25eeb76598d438fab
  cf71d81d835e589353c8404105559ba4759ef05f09f88d9fd2ac559c77a87d52
  8436481d3fdc4834b86ea964d91e15c8bd1dfd478355431765a59c395fbdbd61
  c7eb3a6435715146c473d7d4afdc2b45efa20f11384ed4c6c9526a32e926eac2
  7aa14813f21f6853f6b8966eef0cc496843b1f3d167c897074c8acc2ffb01f2d
  8acbc7128146c673df62a3f3ea5d02d483d7fd29286d189ebb909b7bf6ab42f1
  8ad5eaf84e3c8889cdf42112e17c3bed67b6dc9ab9458709029c5507d44385e9
  3d21f7e576bd4fc64d7c15fd9c0a4df5aa2d831bd000de3bc5d077d523965396
  25911e8186095265b1d6ab55a5d26c56dc984bc6ece113234fd3d1c2771cec18
  c60a3de2c314d73b72a5e4edc05e971838652cb290969120134901ec44685901
)
for index in "${!files[@]}"; do
  observed=$(sha256sum "$here/${files[$index]}" | cut -d' ' -f1)
  test "$observed" = "${hashes[$index]}"
done

"$python" -m py_compile \
  "$here/catvm_growing_rotor_open_momentum_protocol.py" \
  "$service" "$controller" "$oracle"

nice -n 10 ionice -c 3 "$python" "$controller" \
  "$service" "$build_dir/controller-audit" \
  --output "$build_dir/controller.rerun.json"
cmp "$sealed" "$build_dir/controller.rerun.json"

nice -n 10 ionice -c 3 "$python" "$oracle" \
  "$service" "$controller" "$build_dir/oracle-audit" \
  --output "$build_dir/oracle.rerun.json"
cmp "$sealed_oracle" "$build_dir/oracle.rerun.json"

jq -e '
  .classification == "SOURCE_AUDITED_PACKAGE_LOCAL" and
  .verification_level == "PACKAGE_SELF_REVIEW" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .machine_boundary == "SEPARATE_NON_DUMPABLE_NO_NEW_PRIVS_SAME_UID_LINUX_FIXED_BINARY_PIPE_SERVICE" and
  (.controller_imports_or_loads_backend | not) and
  (.controller_computes_boundary | not) and
  .inplace.primary_boundary == 83 and
  .inplace.reuse_boundary == 70 and
  .inplace.fresh_reuse_boundary == 70 and
  .inplace.fresh_restored_reuse_boundary_parity and
  .inplace.fresh_restored_reuse_receipt_parity and
  .inplace.restoration_generation_after_reuse == 2 and
  .inplace.same_service_process_reused and
  .inplace.response_write_after_boundary_retention_and_restoration and
  .inplace.persistent_port_lease_events == 32 and
  .inplace.persistent_port_release_events == 32 and
  .inplace.carrier_field_cells == 8943 and
  .inplace.one_body_terms == 663408 and
  .inplace.process_memory_open_denied and
  ([.controls.resident[] | .denied_without_boundary and .port_resident_before_denial and .port_released_before_restoration and .restored_before_response] | all) and
  ([.controls.inverse[] | .mutation_discriminated and .repair_restored_before_denial_response and (.boundary_released | not)] | all) and
  .controls.snapshot_command_on_inplace_service_denied and
  .controls.null_command_on_live_service_denied and
  .controls.wrong_generation_denied and
  .snapshot_sham.classification == "SNAPSHOT_RELOAD" and
  .snapshot_sham.snapshot_field_cells == 8943 and
  .snapshot_sham.inplace_command_on_snapshot_service_denied and
  .null_carrier.carrier_field_cells == 0 and
  .null_carrier.inplace_command_denied and
  .disconnect.restoration_precedes_disconnected_write_attempt and
  .disconnect.boundary_response_bytes == 0 and
  .disconnect.stderr_bytes == 0 and
  .no_smuggle.audit_event_vocabulary_only and
  (.no_smuggle.audit_contains_intermediate_field_values | not) and
  (.no_smuggle.response_contains_intermediate_cells | not) and
  .no_smuggle.fixed_response_bytes == 44 and
  .resource_accounting.conservative_accepted_named_field_cells_peak == 15774 and
  .resource_accounting.snapshot_sham_conservative_named_field_cells_peak == 24717 and
  .resource_accounting.accepted_retained_transition_plan_entries == 0 and
  .resource_accounting.accepted_retained_inverse_history_bytes == 0 and
  .resource_accounting.accepted_snapshot_cells == 0 and
  .resource_accounting.warm_direct_m199_declared_named_field_cells == 11220 and
  .resource_accounting.warm_direct_m199_forward_plus_inverse_one_body_terms == 663408 and
  .matched_baseline.distinct_phase_resource_observed == false and
  (.not_established | index("COMPUTATIONAL_ADVANTAGE_OR_SMALL_WALL_CROSSING")) != null and
  (.terminal | not)
' "$sealed" >/dev/null

jq -e '
  .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE" and
  .verification_level == "INDEPENDENT_ORACLE_REEXECUTION" and
  .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION" and
  .result == "PASS" and
  (.independence.catvm_service_imported | not) and
  (.independence.catvm_controller_imported | not) and
  (.independence.catvm_protocol_imported | not) and
  (.independence.production_factor_module_imported | not) and
  .independence.separate_m199_mathematical_oracle_reused and
  .algebra.occupation_histograms == 74613 and
  .algebra.necklace_cells == 4389 and
  .algebra.bracelet_cells == 2277 and
  .algebra.primary_boundary == 83 and
  .algebra.reuse_boundary == 70 and
  .algebra.direct_primary_mismatch_cells == 0 and
  .algebra.direct_two_body_terms == 684624 and
  .algebra.direct_two_body_csr_nonzeros == 172838 and
  .algebra.full_unpaired_factor_terms == 494496 and
  .algebra.reflection_paired_factor_terms == 331704 and
  .accepted_protocol.generation == 2 and
  .accepted_protocol.response_order_verified and
  .accepted_protocol.hidden_port_lease_release_pairs == 32 and
  .accepted_protocol.same_service_reused and
  .accepted_protocol.process_memory_open_denied and
  .accepted_protocol.carrier_field_cells == 8943 and
  .accepted_protocol.one_body_terms_per_forward_inverse_transaction == 663408 and
  .fresh_and_snapshot.fresh_matches_restored_reuse and
  .fresh_and_snapshot.snapshot_restoration_classification == "SNAPSHOT_RELOAD" and
  .controls.premature_projection_denied_while_port_resident and
  .controls.missing_inverse_discriminated and
  .controls.wrong_inverse_discriminated and
  .controls.reordered_noncommuting_inverse_discriminated and
  .controls.wrong_reflection_inverse_discriminated and
  .controls.all_controls_restored_before_denial_response and
  .disconnect.restoration_precedes_disconnected_write_attempt and
  .source_boundary_audit.service_response_constructed_after_restoration_verification and
  (.source_boundary_audit.controller_imports_backend_or_service | not) and
  (.source_boundary_audit.controller_dynamically_loads_backend_or_service | not) and
  .source_boundary_audit.service_protocol_write_sites_are_fixed_binary and
  .no_smuggle.audit_event_vocabulary_only and
  (.no_smuggle.intermediate_field_values_in_audit | not) and
  (.no_smuggle.intermediate_cells_in_fixed_response | not)
' "$sealed_oracle" >/dev/null

"$python" - "$controller" "$service" "$oracle" <<'PY'
import ast
import pathlib
import sys

controller, service, oracle = map(lambda item: pathlib.Path(item), sys.argv[1:])

def imports(path):
    result = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module or "")
    return result

controller_imports = imports(controller)
if any("growing_rotor_open_momentum_factor" in name or "catvm_growing_rotor_open_momentum_service" in name for name in controller_imports):
    raise SystemExit("controller backend isolation failed")
oracle_imports = imports(oracle)
if any("catvm_growing_rotor_open_momentum" in name or name == "growing_rotor_open_momentum_factor_closure" for name in oracle_imports):
    raise SystemExit("oracle package isolation failed")
service_text = service.read_text(encoding="utf-8")
block = service_text[service_text.index("    def run_inplace("):service_text.index("    def run_snapshot(")]
tokens = (
    "boundary = backend.boundary",
    "self.verify_restoration",
    "self.generation += 1",
    "response = self.response",
    'self.audit("RESPONSE_WRITE_ATTEMPT")',
)
positions = [block.index(token) for token in tokens]
if positions != sorted(positions):
    raise SystemExit("atomic response ordering source audit failed")
PY

sha256sum "$controller" "$service" "$oracle" "$sealed" "$sealed_oracle"
printf '%s\n' 'QUALIFIED_CATVM_GROWING_ROTOR_OPEN_MOMENTUM_STRICT_SCOPE'
printf 'evidence=managed-disk-backed:%s\n' "$build_dir"
