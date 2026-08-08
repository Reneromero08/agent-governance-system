#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 || ! -d "$1" ]]; then
    echo "usage: $0 DISK_BACKED_BUILD_DIRECTORY" >&2
    exit 2
fi
build=$(realpath -e -- "$1")
case "$build" in
    /dev/shm|/dev/shm/*|/run/shm|/run/shm/*)
        echo "RAM-backed M229 build directory is forbidden" >&2
        exit 2
        ;;
esac
case "$(findmnt -n -o FSTYPE -T "$build")" in
    tmpfs|ramfs)
        echo "RAM-backed M229 build filesystem is forbidden" >&2
        exit 2
        ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production="$here/su2_level8_typed_open_graph_relation.py"
reference="$here/su2_level8_typed_open_graph_relation_separate_reference.py"
sealed_reference="$here/SU2_LEVEL8_TYPED_OPEN_GRAPH_RELATION_SEPARATE_REFERENCE.json"
sealed_result="$here/SU2_LEVEL8_TYPED_OPEN_GRAPH_RELATION_RESULTS.json"
generated_reference="$build/SU2_LEVEL8_TYPED_OPEN_GRAPH_RELATION_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_TYPED_OPEN_GRAPH_RELATION_RESULTS.json"
mkdir -p "$build/tmp" "$build/xdg-cache" "$build/pycache"

env TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp" \
    XDG_CACHE_HOME="$build/xdg-cache" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="$build/pycache" PYTHONPATH="$here" \
    nice -n 10 ionice -c 3 python3 "$reference" > "$generated_reference"
env TMPDIR="$build/tmp" TMP="$build/tmp" TEMP="$build/tmp" \
    XDG_CACHE_HOME="$build/xdg-cache" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX="$build/pycache" PYTHONPATH="$here" \
    nice -n 10 ionice -c 3 python3 "$production" "$generated_reference" \
    > "$generated_result"

cmp "$generated_reference" "$sealed_reference"
cmp "$generated_result" "$sealed_result"

jq -e '
    .result == "PASS_BOUNDED_EXACT_TYPED_SHARED_PORT_REVERSIBLE_MAP_EXECUTION_WITH_IDENTICAL_CLASSICAL_RECURRENCE"
    and .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and .controls.fusion_twist_noncommuting
    and .controls.fusion_cubic_noncommuting
    and .controls.twist_cubic_noncommuting
    and .controls.wrong_owner_rejected
    and .controls.wrong_generation_rejected
    and .controls.stale_generation_rejected
    and .controls.wrong_consumer_rejected
    and .controls.wrong_public_program_rejected
    and .controls.premature_projection_rejected
    and .controls.missing_inverse_detected
    and .controls.reordered_inverse_rejected
    and .controls.wrong_inverse_parameter_changes_restored_state
    and .controls.null_carrier_rejected
    and .controls.wrong_port_type_rejected
    and .controls.semantic_perturbation_changes_state
    and .controls.control_restored_exactly
    and (.controls.public_topology_compilation_reads_final_answer | not)
    and (.controls.relation_tables_materialized | not)
    and (.controls.assignment_expansions_materialized | not)
    and (.cases | length == 4)
    and (.cases | all(
        .module_count == (3 * .rounds)
        and (.distinct_module_kinds | length) == 3
        and .distinct_consumers == [1, 2, 3]
        and .port_type == 22908
        and .same_coefficient_backing
        and .same_scratch_backing
        and .canonical_post_restoration_state_exact
        and .restoration_generation == 1
        and (.baseline_reload_used | not)
        and .forward_payload_bits > 0
        and .work.graph_relation_module_consumptions == .module_count
        and .work.graph_relation_inverse_consumptions == .module_count
        and .work.cubic_shears == .rounds
        and .work.cubic_inverse_shears == .rounds
        and .work.graph_relation_tables_materialized == 0
        and .work.assignment_expansions_materialized == 0
        and .work.maximum_declared_live_payload_bits > 0
    ))
    and .reuse.primary.restoration_generation == 1
    and .reuse.reuse.restoration_generation == 2
    and .reuse.fresh_reuse.restoration_generation == 1
    and .reuse.restoration_generation_after_reuse == 2
    and .reuse.fresh_restored_reuse_boundary_agreement
    and .reuse.fresh_restored_reuse_state_agreement
    and .reuse.primary.same_coefficient_backing
    and .reuse.primary.same_scratch_backing
    and .reuse.reuse.same_coefficient_backing
    and .reuse.reuse.same_scratch_backing
    and .reuse.fresh_reuse.same_coefficient_backing
    and .reuse.fresh_reuse.same_scratch_backing
    and (.reuse.primary.baseline_reload_used | not)
    and (.reuse.reuse.baseline_reload_used | not)
    and (.reuse.fresh_reuse.baseline_reload_used | not)
    and .execution_law.actual_shared_port_cells == 9
    and .execution_law.typed_port == 22908
    and .execution_law.sequential_full_vector_execution
    and .execution_law.distinct_public_consumer_labels == 3
    and (.execution_law.actual_independent_consumers | not)
    and .execution_law.nominal_descriptor_consumer_labels
    and (.execution_law.noncommuting_module_kinds | length) == 3
    and .execution_law.modules_are_exact_reversible_maps
    and (.execution_law.open_relation_signatures_constructed | not)
    and (.execution_law.open_relation_signature_closure | not)
    and (.execution_law.shared_unresolved_relational_port_contraction | not)
    and .execution_law.relation_table_cells == 0
    and .execution_law.assignment_expansions == 0
    and (.execution_law.intermediate_projection | not)
    and .execution_law.final_boundary_only
    and .resource_law.primary_forward_payload_bits > 0
    and .resource_law.primary_maximum_declared_live_payload_bits > 0
    and .resource_law.primary_maximum_declared_live_field_cells > 18
    and .resource_law.retained_public_module_descriptor_integers_counted
    and .resource_law.primary_retained_public_module_descriptor_integer_cells == 63
    and (.resource_law.program_commitment_compilation_transients_complete | not)
    and .resource_law.cubic_transients_counted
    and (.resource_law.inherited_linear_module_scalar_arithmetic_live_payload_complete | not)
    and (.resource_law.whole_transaction_live_payload_complete | not)
    and .matched_classical_baseline.strongest_compact
        == "IDENTICAL_NINE_COORDINATE_FUSION_TWIST_CUBIC_SHEAR_RECURRENCE"
    and .matched_classical_baseline.every_case_state_and_boundary_agreement
    and (.matched_classical_baseline.phase_specific_reduction | not)
    and (.matched_classical_baseline.computational_advantage | not)
    and (.separate_reference.imports_m229_production | not)
    and (.separate_reference.imports_m211_production | not)
    and .separate_reference.uses_independent_polynomial_quotient_substrate
    and .separate_reference.case_and_reuse_semantic_parity
    and (.claim_limits.general_open_relation_algebra | not)
    and (.claim_limits.open_relation_signature_closure | not)
    and (.claim_limits.nonfunctional_multivalued_relational_closure | not)
    and (.claim_limits.catvm_custody | not)
    and (.claim_limits.distinct_phase_resource_established | not)
    and (.claim_limits.computational_advantage | not)
    and (.claim_limits.small_wall_crossed | not)
    and (.claim_limits.physical_waveform_execution | not)
    and (.claim_limits.physical_bit_replacement | not)
    and (.claim_limits.catalytic_inference_established | not)
    and (.claim_limits.unbounded_computation_established | not)
    and (.terminal | not)
' "$generated_result" >/dev/null

jq -e '
    .schema == "cat_cas.su2_level8_typed_open_graph_relation_reference.v1"
    and (.imports_m229_production | not)
    and (.imports_m211_production | not)
    and .uses_independent_polynomial_quotient_substrate
    and (.cases | length == 4)
    and (.cases | all(.canonical_post_restoration_state_exact))
    and .reuse.fresh_restored_reuse_boundary_agreement
    and .reuse.fresh_restored_reuse_state_agreement
    and .reuse.restoration_generation_after_reuse == 2
' "$generated_reference" >/dev/null

python3 - "$reference" <<'PY'
import ast
from pathlib import Path
import sys

tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        imports.add(node.module or "")
for forbidden in (
    "su2_level8_typed_open_graph_relation",
    "root_of_unity_su2_level8_fusion_phase_relation",
):
    if forbidden in imports:
        raise SystemExit(f"M229 separate reference imports production: {forbidden}")
PY

rg -Fq 'phase = su2.K.zeta(module.parameter)' "$production"
rg -Fq 'work.subtract(coefficients[module.target], injected)' "$production"
rg -Fq 'work.add(coefficients[module.target], injected)' "$production"
rg -Fq 'injected = oracle.E.root(module.parameter) * source * source * source' "$reference"
if rg -q 'itertools|CartesianProduct|product\(' "$production" "$reference"; then
    echo "M229 accepted path contains an expansion construct" >&2
    exit 2
fi

echo "QUALIFIED_SU2_LEVEL8_TYPED_OPEN_GRAPH_RELATION_STRICT_SCOPE"
