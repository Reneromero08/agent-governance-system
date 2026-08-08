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

production_cpp="$here/su2_level8_markov_skein_krylov_core.cpp"
reference_cpp="$here/su2_level8_markov_skein_krylov_separate_reference.cpp"
wrapper="$here/su2_level8_markov_skein_krylov.py"
production_bin="$build/su2_level8_markov_skein_krylov_core"
reference_bin="$build/su2_level8_markov_skein_krylov_reference"
generated_reference="$build/SU2_LEVEL8_MARKOV_SKEIN_KRYLOV_SEPARATE_REFERENCE.json"
generated_result="$build/SU2_LEVEL8_MARKOV_SKEIN_KRYLOV_RESULTS.json"

g++ -std=c++20 -O3 -Wall -Wextra -Werror -pedantic \
    "$production_cpp" -o "$production_bin"
g++ -std=c++20 -O3 -Wall -Wextra -Werror -pedantic \
    "$reference_cpp" -o "$reference_bin"

nice -n 10 ionice -c 2 -n 7 "$reference_bin" > "$generated_reference"
PYTHONPATH="$here" nice -n 10 ionice -c 2 -n 7 python3 \
    "$wrapper" "$production_bin" "$reference_bin" > "$generated_result"

cmp "$generated_reference" \
    "$here/SU2_LEVEL8_MARKOV_SKEIN_KRYLOV_SEPARATE_REFERENCE.json"
cmp "$generated_result" "$here/SU2_LEVEL8_MARKOV_SKEIN_KRYLOV_RESULTS.json"

jq -e '
    .classification == "INDEPENDENTLY_VERIFIED_STRICT_SCOPE"
    and .verification_level == "SEPARATE_REFERENCE_PARITY"
    and .restoration_classification == "EXACT_ALGEBRAIC_RESTORATION"
    and .degree_law.all_cases_full_at_both_primes
    and .degree_law.all_primary_markov_sequences_identical_to_m217_vacuum_sequences
    and .representation_parity.exact_primary_boundary_identical_to_m217
    and .controls.alternate_public_program_family_boundary_parity
    and .separate_reference.reference_imports_m218_production == false
    and .separate_reference.all_cases_full_at_both_distinct_split_primes
    and .transaction.primary.canonical_post_restoration_state_exact
    and .transaction.primary.same_coefficient_backing
    and .transaction.primary.same_scratch_backing
    and .transaction.reuse.canonical_post_restoration_state_exact
    and .transaction.fresh_restored_reuse_boundary_agreement
    and .transaction.fresh_restored_reuse_state_agreement
    and .claim_limits.jones_wenzl_reduced_diagram_carrier_established == false
    and .claim_limits.distinct_phase_resource_established == false
    and .claim_limits.computational_advantage == false
    and .terminal == false
' "$generated_result" >/dev/null

if rg -q 'su2_level8_markov_skein_krylov_(core|production)|#include "su2_level8_markov_skein_krylov' \
    "$reference_cpp"; then
    echo "separate reference imports M218 production" >&2
    exit 2
fi

PYTHONPYCACHEPREFIX="$build/pycache" python3 -m py_compile "$wrapper"

echo "QUALIFIED_SU2_LEVEL8_MARKOV_SKEIN_FULL_BOUNDARY_DEGREE_OBSTRUCTION_STRICT_SCOPE"
