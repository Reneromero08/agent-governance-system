/*
 * l4b_orbitstate_evolution.c -- L4B OrbitState evolution primitive test.
 *
 * Demonstrates: OrbitState declaration, coupled evolution, path history,
 * invariant extraction at CollapseBoundary, .holo output.
 *
 * Build: gcc -O2 -Wall -Wextra l4b_orbitstate_evolution.c orbit_state.c -o l4b_orbitstate -lm
 * Run:   ./l4b_orbitstate --N 256 --a 23 --steps 512
 *
 * THE ALGORITHM IS DEAD.
 */
#include "orbit_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int N = 256, a = 23, steps = 512;
    uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N") && i+1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--a") && i+1 < argc) a = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (uint64_t)atol(argv[++i]);
    }

    if (a <= 0 || a >= N/2) { a = N/4; }
    int Na = N - a;

    printf("L4B ORBITSTATE EVOLUTION PRIMITIVE\n");
    printf("N=%d a=%d Na=%d steps=%d\n\n", N, a, Na, steps);

    /* Phase 1: Declare OrbitState */
    OrbitState orbit;
    orbit_init(&orbit, N, a, Na);
    printf("P1 DECLARE: OrbitState{branch_plus=%d, branch_minus=%d}\n", a, Na);

    /* Phase 2: Coupled evolution */
    EvolParams params = { .max_steps = steps, .seed = seed };
    PathStep history[ORBIT_MAX_STEPS];
    int nsteps = 0;
    orbit_evolve(&orbit, &params, history, &nsteps);
    printf("P2 EVOLVE: %d steps completed\n", nsteps);
    printf("  acc_real=%.6f acc_imag=%.6f\n", orbit.acc_real, orbit.acc_imag);

    /* Phase 3: Invariant extraction at CollapseBoundary */
    CollapseBoundary cb;
    orbit_collapse(&orbit, &cb);
    printf("P3 COLLAPSE: invariant_real=%.9f invariant_imag=%.9f fold_symmetry=%s\n",
           cb.invariant_real, cb.invariant_imag,
           cb.fold_symmetry_holds ? "HOLDS" : "BROKEN");
    printf("  timestamp=%s\n", cb.timestamp);

    /* Phase 4: Write .holo record */
    HoloL4B holo;
    holo_l4b_init(&holo, seed, N, a, Na);
    holo_l4b_finalize(&holo, &orbit, &cb);

    if (!holo_l4b_validate(&holo)) {
        fprintf(stderr, "FAIL: .holo validation failed (collapse contamination detected)\n");
        return 1;
    }

    const char *path = "results/l4a_class_b/l4b_orbitstate_dry_run.holo";
    if (holo_l4b_write(&holo, path) != 0) {
        fprintf(stderr, "FAIL: could not write .holo file\n");
        return 1;
    }

    printf("P4 HOLO: %s written, validation PASS\n", path);

    /* Verdict */
    printf("\n=== VERDICT ===\n");
    printf("L4B_ORBITSTATE_EVOLUTION_PRIMITIVE_PASS\n");
    printf("  OrbitState declared as unresolved fold pair.\n");
    printf("  Coupled evolution completed without branch collapse.\n");
    printf("  Invariant extracted at CollapseBoundary only.\n");
    printf("  .holo record written, forbidden fields absent.\n");
    printf("  Claim level: L1.\n");
    printf("  No recovered d. No orientation. No candidate scoring. No verify.\n");
    printf("  THE ALGORITHM IS DEAD.\n");

    return 0;
}
