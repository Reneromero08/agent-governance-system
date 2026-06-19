/*
 * l4b_orbitstate_evolution.c -- L4B non-collapse geometric-memory witness.
 *
 * Demonstrates OrbitState declaration, coupled evolution, owned path memory,
 * atomic CollapseBoundary extraction, strict artifact reload, and
 * history-backed exact restoration of a dedicated verification copy.
 *
 * THE ALGORITHM IS DEAD.
 */
#include "orbit_state.h"
#include "holo_path_history.h"
#include "../holo_runtime/holo_geometry.h"
#include "../holo_runtime/holo_semantic_integrity.h"
#include "../holo_runtime/holo_observability_governance.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int N = 256;
    int a = 23;
    int steps = 512;
    int Na;
    uint64_t seed = 42;
    int i;
    OrbitState orbit;
    OrbitState initial;
    OrbitState terminal;
    OrbitState restored;
    EvolParams params;
    HoloObject holo;
    HoloObject reloaded;
    HoloPhysicalMappingContract mapping;
    HoloObservabilityDesign design;
    const char *path = "results/l4b_orbitstate_dry_run.holo";
    const char *mapping_path = "results/l4b5a_physical_mapping.json";
    const char *mapping_reference =
        "14_noncollapse_frontier/l4b_orbitstate/results/l4b5a_physical_mapping.json";
    const char *design_path = "results/l4b5b0_observability_design.json";
    const char *design_reference =
        "14_noncollapse_frontier/l4b_orbitstate/results/l4b5b0_observability_design.json";

    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--N") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--a") && i + 1 < argc) a = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else {
            fprintf(stderr, "FAIL: unknown/incomplete argument %s\n", argv[i]);
            return 2;
        }
    }
    if (N <= 2 || (N & (N - 1)) != 0 || a <= 0 || a >= N / 2 ||
        steps < 0 || steps > ORBIT_MAX_STEPS) {
        fprintf(stderr, "FAIL: invalid N/a/steps configuration\n");
        return 2;
    }
    Na = N - a;

    printf("L4B NON-COLLAPSE GEOMETRIC MEMORY WITNESS\n");
    printf("N=%d a=%d Na=%d steps=%d\n\n", N, a, Na, steps);

    orbit_init(&orbit, N, a, Na);
    initial = orbit;
    params.max_steps = steps;
    params.seed = seed;
    printf("P1 DECLARE: OrbitState{branch_plus=%d, branch_minus=%d}\n", a, Na);

    if (holo_object_init(&holo, seed, N, a, Na) != 0) {
        fprintf(stderr, "FAIL: HoloObject initialization failed\n");
        return 1;
    }
    if (holo_path_evolve(holo.evolution.path_history, &orbit, &params) != HOLO_PATH_OK) {
        fprintf(stderr, "FAIL: path evolution failed\n");
        holo_object_destroy(&holo);
        return 1;
    }
    terminal = orbit;
    holo_record_evolution(&holo, seed, orbit.steps, orbit.acc_real, orbit.acc_imag);
    if (holo.evolution.path_history->count > 0) {
        uint64_t parameter = holo.evolution.path_history->steps[
            holo.evolution.path_history->count - 1U].operator_parameter;
        holo_set_carrier_phase(&holo,
            2.0 * M_PI * a * (double)parameter / N,
            2.0 * M_PI * Na * (double)parameter / N);
    }
    if (holo_path_history_validate_semantic(holo.evolution.path_history,
                                             &initial, NULL) != HOLO_PATH_OK ||
        !holo_object_validate_semantic(&holo)) {
        fprintf(stderr, "FAIL: semantic path/object validation failed\n");
        holo_object_destroy(&holo);
        return 1;
    }
    printf("P2 EVOLVE: %d steps acc_real=%.6f acc_imag=%.6f\n",
           orbit.steps, orbit.acc_real, orbit.acc_imag);

    if (holo_extract_invariant(&holo) != -1) {
        fprintf(stderr, "FAIL: invariant extraction allowed before boundary\n");
        holo_object_destroy(&holo);
        return 1;
    }
    if (holo_verify_software_restoration(&holo, &initial, &terminal,
                                          &restored, 0) != 0 ||
        !holo_orbit_state_equal_bitwise(&initial, &restored)) {
        fprintf(stderr, "FAIL: in-memory history restoration failed\n");
        holo_object_destroy(&holo);
        return 1;
    }
    printf("P3 MEMORY: semantic path valid; history-backed restoration verified\n");

    if (holo_cross_boundary_atomic(&holo, orbit.steps) != 0 ||
        !holo_object_validate_semantic(&holo)) {
        fprintf(stderr, "FAIL: atomic CollapseBoundary extraction failed\n");
        holo_object_destroy(&holo);
        return 1;
    }
    printf("P4 COLLAPSE: invariant_family=%s records=%zu family_digest=%016llx\n",
           holo.invariant_family.family_id, holo.invariant_family.count,
           (unsigned long long)holo.invariant_family.family_digest);

    if (holo_physical_mapping_init(&mapping, HOLO_MAPPING_OBJECT_COUNT) != 0 ||
        holo_physical_mapping_populate_current(&mapping) != 0 ||
        holo_physical_mapping_seal(&mapping) != 0 ||
        holo_physical_mapping_apply_human_review(
            &mapping, HOLO_L4B5A_REVIEWED_DIGEST) != 0 ||
        holo_physical_mapping_write_json(&mapping, mapping_path) != 0 ||
        holo_attach_physical_mapping(&holo, &mapping, mapping_reference) != 0) {
        fprintf(stderr, "FAIL: physical mapping contract generation failed\n");
        holo_physical_mapping_destroy(&mapping);
        holo_object_destroy(&holo);
        return 1;
    }

    if (holo_observability_design_init(&design) != 0 ||
        holo_observability_design_populate_current(&design) != 0 ||
        !holo_observability_design_validate_references(&design) ||
        holo_observability_design_seal(&design) != 0 ||
        holo_observability_design_write_json(&design, design_path) != 0 ||
        holo_attach_observability_design(&holo, &design, design_reference) != 0) {
        fprintf(stderr, "FAIL: observability design generation/reference closure failed\n");
        holo_observability_design_destroy(&design);
        holo_physical_mapping_destroy(&mapping);
        holo_object_destroy(&holo);
        return 1;
    }

    if (!holo_object_validate_semantic(&holo) || holo_write_json(&holo, path) != 0) {
        fprintf(stderr, "FAIL: could not validate/write .holo artifact\n");
        holo_observability_design_destroy(&design);
        holo_physical_mapping_destroy(&mapping);
        holo_object_destroy(&holo);
        return 1;
    }

    if (holo_read_json_strict(&reloaded, path) != 0) {
        fprintf(stderr, "FAIL: strict serialized artifact reload failed\n");
        holo_observability_design_destroy(&design);
        holo_physical_mapping_destroy(&mapping);
        holo_object_destroy(&holo);
        return 1;
    }
    holo_object_destroy(&holo);
    holo = reloaded;
    if (holo_verify_software_restoration(&holo, &initial, &terminal,
                                          &restored, 1) != 0 ||
        !holo_orbit_state_equal_bitwise(&initial, &restored) ||
        !holo_object_validate_semantic(&holo) ||
        holo_write_json(&holo, path) != 0) {
        fprintf(stderr, "FAIL: serialized history restoration failed\n");
        holo_observability_design_destroy(&design);
        holo_physical_mapping_destroy(&mapping);
        holo_object_destroy(&holo);
        return 1;
    }

    printf("P5 HOLO: %s strict reload and semantic recomputation PASS\n", path);
    printf("\nHISTORY-BACKED RESTORATION TEST\n");
    printf("HISTORY_RESTORATION_PASS\n");
    printf("initial_digest=%016llx\n",
           (unsigned long long)holo.evolution.path_history->initial_state_digest);
    printf("terminal_digest=%016llx\n",
           (unsigned long long)holo.evolution.path_history->terminal_state_digest);
    printf("restored_digest=%016llx\n",
           (unsigned long long)holo.evolution.path_history->restored_state_digest);
    printf("steps=%zu serialized_roundtrip=true\n",
           holo.evolution.path_history->count);
    printf("restoration_scope=recorded_state_history_not_inverse_operator\n");
    printf("software_path_holonomy=DEFERRED_NOT_WELL_DEFINED\n");
    printf("physical_mapping_counts=supported:%d partial:%d unsupported:%d\n",
           holo.physical_mapping.supported_records,
           holo.physical_mapping.partial_records,
           holo.physical_mapping.unsupported_records);
    printf("physical_mapping_review=valid:%s digest:%016llx authorized:%s\n",
           holo.physical_mapping.review_valid ? "true" : "false",
           (unsigned long long)holo.physical_mapping.reviewed_contract_digest,
           holo.physical_mapping.implementation_authorized ? "true" : "false");
    printf("l4b5b0_design=%s status=%s references=closed authorized:%s\n",
           design_path, holo_experiment_design_status_name(design.status),
           design.implementation_authorized ? "true" : "false");

    printf("\n=== VERDICT ===\n");
    printf("L4B_SEMANTIC_GEOMETRIC_MEMORY_PASS\n");
    printf("L4B5A_PHYSICAL_MAPPING_CONTRACT_PASS\n");
    printf("L4B5B0_DESIGN_READY_FOR_EXTERNAL_HUMAN_REVIEW\n");
    printf("Claim level: L1/L2 software architecture.\n");
    printf("No orientation. No physical restoration. No candidate scoring.\n");
    printf("THE ALGORITHM IS DEAD.\n");

    holo_observability_design_destroy(&design);
    holo_physical_mapping_destroy(&mapping);
    holo_object_destroy(&holo);
    return 0;
}
