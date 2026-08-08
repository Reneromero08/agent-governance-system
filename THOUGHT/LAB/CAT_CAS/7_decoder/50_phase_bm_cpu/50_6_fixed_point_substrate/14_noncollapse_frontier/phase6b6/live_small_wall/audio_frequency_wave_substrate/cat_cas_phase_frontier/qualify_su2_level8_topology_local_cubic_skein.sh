#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 DISK_BACKED_BUILD_DIRECTORY" >&2
    exit 2
fi

build=$1
case "$build" in
    /dev/shm/*|/run/shm/*)
        echo "RAM-backed build directory is forbidden" >&2
        exit 2
        ;;
esac

here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$build"

production="$here/su2_level8_topology_local_cubic_skein.py"
reference="$here/su2_level8_topology_local_cubic_skein_separate_reference.py"
generated_reference="$build/SU2_LEVEL8_TOPOLOGY_LOCAL_CUBIC_SKEIN_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_TOPOLOGY_LOCAL_CUBIC_SKEIN_RESULTS.json"

PYTHONPYCACHEPREFIX="$build/pycache" nice -n 10 ionice -c 2 -n 7 \
    python3 "$reference" > "$generated_reference"
PYTHONPATH="$here" PYTHONPYCACHEPREFIX="$build/pycache" \
    nice -n 10 ionice -c 2 -n 7 \
    python3 "$production" "$generated_reference" > "$generated_result"

cmp "$generated_reference" \
    "$here/SU2_LEVEL8_TOPOLOGY_LOCAL_CUBIC_SKEIN_SEPARATE_REFERENCE.json"
cmp "$generated_result" \
    "$here/SU2_LEVEL8_TOPOLOGY_LOCAL_CUBIC_SKEIN_RESULTS.json"

jq -e '
    .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and .degree_law.all_declared_cases_match_formula
    and .separate_reference.imports_m219_or_m218_production == false
    and .separate_reference.exact_height_tuple_parity
    and .separate_reference.exact_state_and_boundary_commitment_parity
    and .separate_reference.generic_leading_degree_parity
    and .transaction.primary.canonical_post_restoration_state_exact
    and .transaction.primary.same_coefficient_backing
    and .transaction.primary.same_scratch_backing
    and .transaction.reuse.canonical_post_restoration_state_exact
    and .transaction.fresh_restored_reuse_boundary_agreement
    and .transaction.fresh_restored_reuse_state_agreement
    and .controls.wrong_shear_phase_detected
    and .controls.missing_shear_inverse_detected
    and .controls.skein_and_cubic_shear_noncommute
    and (.controls.intermediate_link_pattern_state_projected | not)
    and (.controls.snapshot_command_available | not)
    and (.claim_limits.compact_exact_height_closure_established | not)
    and (.claim_limits.distinct_phase_resource_established | not)
    and (.claim_limits.computational_advantage | not)
    and (.terminal | not)
' "$generated_result" >/dev/null

if rg -q '^(from|import) su2_level8_(topology_local_cubic_skein|markov_skein_krylov|fusion_path_braid_phase_relation)' \
    "$reference"; then
    echo "separate reference imports a CAT_CAS production module" >&2
    exit 2
fi

PYTHONPYCACHEPREFIX="$build/pycache" python3 -m py_compile \
    "$production" "$reference"

echo "QUALIFIED_SU2_LEVEL8_TOPOLOGY_LOCAL_CUBIC_SKEIN_HEIGHT_OBSTRUCTION_STRICT_SCOPE"
