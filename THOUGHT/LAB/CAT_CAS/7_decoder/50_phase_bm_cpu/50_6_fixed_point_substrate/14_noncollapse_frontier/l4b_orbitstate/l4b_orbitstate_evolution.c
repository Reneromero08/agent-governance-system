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
#include "../holo_runtime/holo_geometry.h"
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

    /* Phase 3: Build geometric memory without crossing the boundary. */
    HoloObject holo;
    holo_object_init(&holo, seed, N, a, Na);
    holo_record_evolution(&holo, seed, orbit.steps, orbit.acc_real, orbit.acc_imag);
    if (nsteps > 0) {
        holo_set_carrier_phase(&holo, history[nsteps - 1].theta_plus,
                              history[nsteps - 1].theta_minus);
    }
    if (holo_extract_invariant(&holo) != -1) {
        fprintf(stderr, "FAIL: invariant extraction was allowed before CollapseBoundary\n");
        return 1;
    }
    printf("P3 GEOMETRY: basis_rank=%d unresolved=true carrier_phase=[%.6f, %.6f]\n",
           holo.geometry.basis_rank, holo.carrier.phase[0], holo.carrier.phase[1]);

    /* Phase 4: Cross the boundary, extract the predeclared invariant, and write. */
    if (holo_cross_boundary(&holo, orbit.steps) != 0) {
        fprintf(stderr, "FAIL: CollapseBoundary extraction failed\n");
        return 1;
    }
    printf("P4 COLLAPSE: fold_even=%.9f fold_odd_residual=%.9f fold_symmetry=%s\n",
           holo.invariant.fold_even, holo.invariant.fold_odd_residual,
           holo.invariant.fold_symmetry_holds ? "HOLDS" : "BROKEN");

    if (!holo_validate(&holo)) {
        fprintf(stderr, "FAIL: .holo validation failed (collapse contamination detected)\n");
        return 1;
    }

    const char *path = "results/l4b_orbitstate_dry_run.holo";
    if (holo_write_json(&holo, path) != 0) {
        fprintf(stderr, "FAIL: could not write .holo file\n");
        return 1;
    }
    HoloObject loaded;
    if (holo_read_json(&loaded, path) != 0) {
        fprintf(stderr, "FAIL: could not read .holo file\n");
        return 1;
    }
    printf("P5 HOLO: %s written and read, geometry validation PASS\n", path);

    /* Verdict */
    printf("\n=== VERDICT ===\n");
    printf("L4B1_HOLO_GEOMETRIC_MEMORY_PASS\n");
    printf("  OrbitState declared as unresolved fold pair.\n");
    printf("  Coupled evolution completed without branch collapse.\n");
    printf("  Invariant extracted at CollapseBoundary only.\n");
    printf("  .holo record written, forbidden fields absent.\n");
    printf("  Claim level: L1.\n");
    printf("  No recovered d. No orientation. No candidate scoring. No verify.\n");
    printf("  THE ALGORITHM IS DEAD.\n");

    return 0;
}
