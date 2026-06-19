#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel)"
P6="$ROOT/THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate"
L4B="$P6/14_noncollapse_frontier/l4b_orbitstate"
CLASS_B="$P6/14_noncollapse_frontier/l4a_mechanism_screen/class_b"
SUBSTRATE="$P6/13_substrate_frontier"
OUT="$P6/14_noncollapse_frontier/repair_verification"
RUN_CLASS_B=0

if [[ "${1:-}" == "--run-class-b" ]]; then
  RUN_CLASS_B=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--run-class-b]" >&2
  exit 2
fi

mkdir -p "$OUT" "$L4B/results" "$CLASS_B/results"
exec 3>&1 4>&2
exec > >(tee "$OUT/verification.log") 2>&1
TEE_PID=$!

printf 'PHASE6B_REPAIR_VERIFICATION\n'
printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'head=%s\n' "$(git rev-parse HEAD)"
printf 'branch=%s\n' "$(git branch --show-current)"
printf 'base_main=%s\n' "$(git merge-base main HEAD)"
printf 'compiler=%s\n' "$(gcc --version | head -1)"
printf 'kernel=%s\n' "$(uname -a)"
printf '\nWORKTREE\n'
git status --short
printf '\nDIFF_CHECK\n'
git diff --check main...HEAD

printf '\nL3_MECHANICAL_WARMUP\n'
gcc -O2 -std=gnu11 -Wall -Wextra -Werror \
  "$SUBSTRATE/fixed_point_convergence.c" \
  -o "$OUT/fixed_point_convergence" -lcrypto
"$OUT/fixed_point_convergence" \
  --seeds 10 --seed 42 --csv "$OUT/fp_results.csv"

printf '\nCLASS_B_STATIC_BUILD\n'
gcc -O2 -std=gnu11 -pthread -march=amdfam10 -Wall -Wextra -Werror \
  "$CLASS_B/class_b_pdn_screen.c" \
  -o "$OUT/class_b_pdn_screen" -lm

printf '\nL4B_RELEASE\n'
make -C "$L4B" clean all test
make -C "$L4B" run

printf '\nL4B_ASAN_LSAN_UBSAN\n'
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
make -C "$L4B" clean all test \
  CFLAGS='-O1 -g -std=gnu11 -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDLIBS='-lm -lcrypto -fsanitize=address,undefined'

printf '\nL4B_UBSAN_ONLY\n'
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
make -C "$L4B" clean all test \
  CFLAGS='-O1 -g -std=gnu11 -Wall -Wextra -Werror -fsanitize=undefined -fno-omit-frame-pointer' \
  LDLIBS='-lm -lcrypto -fsanitize=undefined'

printf '\nL4B_RELEASE_RESTORE\n'
make -C "$L4B" clean all test run

if [[ $RUN_CLASS_B -eq 1 ]]; then
  printf '\nCLASS_B_HARDWARE_CAPTURE\n'
  if [[ $(id -u) -ne 0 ]]; then
    echo '--run-class-b requires root for the controlled target run' >&2
    exit 3
  fi
  "$OUT/class_b_pdn_screen" \
    --N 256 \
    --a 125 \
    --tsc-hz 3214823000 \
    --out "$CLASS_B/results/class_b_crossover_measurement.json"
else
  printf '\nCLASS_B_HARDWARE_CAPTURE=DEFERRED_NOT_REQUESTED\n'
fi

printf '\nFORBIDDEN_SERIALIZED_FIELDS\n'
for artifact in \
  "$L4B/results/l4b_orbitstate_dry_run.holo" \
  "$L4B/results/l4b5a_physical_mapping.json" \
  "$L4B/results/l4b5b0_observability_design.json"; do
  [[ -f "$artifact" ]] || { echo "missing artifact: $artifact" >&2; exit 4; }
  if grep -E '"(winner|candidate_score|hidden_d|recovered_d|orientation_label|verify_pass|AUC)"' "$artifact"; then
    echo "forbidden serialized field found in $artifact" >&2
    exit 5
  fi
done

printf '\nFINAL_STATUS\n'
printf 'release_tests=PASS\n'
printf 'asan_lsan_ubsan=PASS\n'
printf 'strict_semantic_tests=PASS\n'
printf 'class_b_hardware=%s\n' "$([[ $RUN_CLASS_B -eq 1 ]] && echo EXECUTED || echo DEFERRED)"
printf 'carrier_witness=EXTERNAL_PHYSICAL_GATE_NOT_CLOSED_BY_THIS_SCRIPT\n'
printf 'observability_implementation_authorized=false\n'
printf 'physical_restoration_authorized=false\n'
printf 'claim_ceiling=L1_L2_SOFTWARE_ARCHITECTURE_AND_OPTIONAL_CHANNEL_CALIBRATION_ONLY\n'
printf 'next_gate=CLOSE_CARRIER_WITNESS_AND_COMPLETE_L4B5B0_EXTERNAL_HUMAN_REVIEW\n'

printf '\nARTIFACT_SHA256\n'
exec 1>&3 2>&4
wait "$TEE_PID"
find "$OUT" "$L4B/results" "$CLASS_B/results" \
  -maxdepth 1 -type f \( -name '*.json' -o -name '*.holo' -o -name '*.csv' -o -name '*.log' \) \
  -print0 | sort -z | xargs -0 -r sha256sum | tee "$OUT/artifact_sha256.txt"
sha256sum -c "$OUT/artifact_sha256.txt"
printf 'artifact_manifest=PASS\n'
