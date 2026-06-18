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
#include "holo_path_history.h"
#include "../holo_runtime/holo_geometry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    printf("L4B.4 NON-COLLAPSE INVARIANT FAMILY\n");
    printf("N=%d a=%d Na=%d steps=%d\n\n", N, a, Na, steps);

    /* Phase 1: Declare OrbitState */
    OrbitState orbit;
    orbit_init(&orbit, N, a, Na);
    printf("P1 DECLARE: OrbitState{branch_plus=%d, branch_minus=%d}\n", a, Na);

    /* Phase 2: Coupled evolution */
    EvolParams params = { .max_steps = steps, .seed = seed };
    OrbitState initial = orbit;
    HoloObject holo;
    if (holo_object_init(&holo, seed, N, a, Na) != 0) {
        fprintf(stderr, "FAIL: HoloObject initialization failed\n");
        return 1;
    }
    if (holo_path_evolve(holo.evolution.path_history, &orbit, &params) != HOLO_PATH_OK) {
        fprintf(stderr, "FAIL: reversible path evolution failed\n");
        holo_object_destroy(&holo);
        return 1;
    }
    OrbitState terminal = orbit;
    printf("P2 EVOLVE: %d steps completed\n", orbit.steps);
    printf("  acc_real=%.6f acc_imag=%.6f\n", orbit.acc_real, orbit.acc_imag);

    /* Phase 3: Build geometric memory without crossing the boundary. */
    holo_record_evolution(&holo, seed, orbit.steps, orbit.acc_real, orbit.acc_imag);
    if (holo.evolution.path_history->count > 0) {
        uint64_t parameter = holo.evolution.path_history->steps[
            holo.evolution.path_history->count - 1U].operator_parameter;
        holo_set_carrier_phase(&holo,
            2.0 * M_PI * a * (double)parameter / N,
            2.0 * M_PI * Na * (double)parameter / N);
    }
    if (holo_extract_invariant(&holo) != -1) {
        fprintf(stderr, "FAIL: invariant extraction was allowed before CollapseBoundary\n");
        holo_object_destroy(&holo);
        return 1;
    }
    printf("P3 GEOMETRY: basis_rank=%d unresolved=true carrier_phase=[%.6f, %.6f]\n",
           holo.geometry.basis_rank, holo.carrier.phase[0], holo.carrier.phase[1]);

    /* Reverse only a verification copy; preserve the terminal state for projection. */
    OrbitState restored;
    if (holo_verify_software_restoration(&holo, &initial, &terminal, &restored, 0) != 0) {
        fprintf(stderr, "FAIL: in-memory path restoration failed\n");
        holo_object_destroy(&holo);
        return 1;
    }

    /* Phase 4: Cross the boundary, seal history, extract, and write. */
    if (holo_cross_boundary(&holo, orbit.steps) != 0) {
        fprintf(stderr, "FAIL: CollapseBoundary extraction failed\n");
        holo_object_destroy(&holo);
        return 1;
    }
    printf("P4 COLLAPSE: invariant_family=%s records=%zu family_digest=%016llx\n",
           holo.invariant_family.family_id, holo.invariant_family.count,
           (unsigned long long)holo.invariant_family.family_digest);

    if (!holo_validate(&holo)) {
        fprintf(stderr, "FAIL: .holo validation failed (collapse contamination detected)\n");
        holo_object_destroy(&holo);
        return 1;
    }

    const char *path = "results/l4b_orbitstate_dry_run.holo";
    const char *mapping_path = "results/l4b5a_physical_mapping.json";
    const char *mapping_reference =
        "14_noncollapse_frontier/l4b_orbitstate/results/l4b5a_physical_mapping.json";
    HoloPhysicalMappingContract mapping;
    if (holo_physical_mapping_init(&mapping, HOLO_MAPPING_OBJECT_COUNT) != 0 ||
        holo_physical_mapping_populate_current(&mapping) != 0 ||
        holo_physical_mapping_seal(&mapping) != 0 ||
        holo_physical_mapping_write_json(&mapping, mapping_path) != 0 ||
        holo_attach_physical_mapping(&holo, &mapping, mapping_reference) != 0) {
        fprintf(stderr, "FAIL: physical mapping contract generation failed\n");
        holo_physical_mapping_destroy(&mapping);
        holo_object_destroy(&holo);
        return 1;
    }
    if (holo_write_json(&holo, path) != 0) {
        fprintf(stderr, "FAIL: could not write .holo file\n");
        holo_physical_mapping_destroy(&mapping);
        holo_object_destroy(&holo);
        return 1;
    }

    /* Reload the whole object; the reader recomputes rather than trusting records. */
    HoloObject reloaded;
    if (holo_read_json(&reloaded, path) != 0) {
        fprintf(stderr, "FAIL: serialized invariant family reload failed\n");
        holo_physical_mapping_destroy(&mapping);
        holo_object_destroy(&holo);
        return 1;
    }
    holo_object_destroy(&holo);
    holo = reloaded;
    if (holo_verify_software_restoration(&holo, &initial, &terminal, &restored, 1) != 0 ||
        holo_write_json(&holo, path) != 0) {
        fprintf(stderr, "FAIL: serialized path restoration failed\n");
        holo_physical_mapping_destroy(&mapping);
        holo_object_destroy(&holo);
        return 1;
    }
    printf("P5 HOLO: %s family reloaded, recomputed, and rewritten\n", path);
    printf("\nPATH REVERSIBILITY TEST\n");
    printf("PATH_REVERSIBILITY_PASS\n");
    printf("initial_state={N:%d,lower:%d,mirror:%d,acc_real:%.17g,acc_imag:%.17g,steps:%d}\n",
           initial.N, initial.branch_plus, initial.branch_minus,
           initial.acc_real, initial.acc_imag, initial.steps);
    printf("terminal_state={N:%d,lower:%d,mirror:%d,acc_real:%.17g,acc_imag:%.17g,steps:%d}\n",
           terminal.N, terminal.branch_plus, terminal.branch_minus,
           terminal.acc_real, terminal.acc_imag, terminal.steps);
    printf("restored_state={N:%d,lower:%d,mirror:%d,acc_real:%.17g,acc_imag:%.17g,steps:%d}\n",
           restored.N, restored.branch_plus, restored.branch_minus,
           restored.acc_real, restored.acc_imag, restored.steps);
    printf("initial_digest=%016llx\n", (unsigned long long)holo.evolution.path_history->initial_state_digest);
    printf("terminal_digest=%016llx\n", (unsigned long long)holo.evolution.path_history->terminal_state_digest);
    printf("restored_digest=%016llx\n", (unsigned long long)holo.evolution.path_history->restored_state_digest);
    printf("steps=%zu\n", holo.evolution.path_history->count);
    printf("serialized_history=%s\n", path);
    printf("reloaded_history_count=%zu\n", holo.evolution.path_history->count);
    printf("equality=bitwise_numeric_orbit_state\n");
    printf("serialized_roundtrip=true\n");
    printf("invariant_family_digest=%016llx\n",
           (unsigned long long)holo.invariant_family.family_digest);
    printf("serialization_invariance=%s\n",
           holo.invariant_family.records[HOLO_INV_SERIALIZATION].passed ? "PASS" : "FAIL");
    printf("software_path_holonomy=DEFERRED_NOT_WELL_DEFINED\n");
    printf("physical_mapping_contract=%s\n", mapping_path);
    printf("physical_mapping_counts=supported:%d partial:%d unsupported:%d\n",
           holo.physical_mapping.supported_records,
           holo.physical_mapping.partial_records,
           holo.physical_mapping.unsupported_records);
    printf("l4b5b_decision=NOT_AUTHORIZED_EVIDENCE_MISSING\n");

    /* Verdict */
    printf("\n=== VERDICT ===\n");
    printf("L4B4_NONCOLLAPSE_INVARIANT_FAMILY_PASS\n");
    printf("L4B5A_PHYSICAL_MAPPING_CONTRACT_PASS\n");
    printf("  OrbitState declared as unresolved fold pair.\n");
    printf("  Coupled evolution completed without branch collapse.\n");
    printf("  Predeclared invariant family extracted at CollapseBoundary only.\n");
    printf("  .holo record written, forbidden fields absent.\n");
    printf("  Serialized path reloaded and initial OrbitState restored bitwise.\n");
    printf("  Claim level: L1.\n");
    printf("  No recovered d. No orientation. No candidate scoring. No verify.\n");
    printf("  THE ALGORITHM IS DEAD.\n");

    holo_physical_mapping_destroy(&mapping);
    holo_object_destroy(&holo);
    return 0;
}
