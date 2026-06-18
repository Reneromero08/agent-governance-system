#include "orbit_state.h"
#include "holo_path_history.h"
#include "../holo_runtime/holo_geometry.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    HoloObject object;
    OrbitState initial;
    OrbitState terminal;
    OrbitState restored;
} Fixture;

static void fixture_evolve(Fixture *fixture, int steps) {
    EvolParams params = { .max_steps = steps, .seed = 42 };
    orbit_init(&fixture->initial, 256, 23, 233);
    fixture->terminal = fixture->initial;
    assert(holo_object_init(&fixture->object, 42, 256, 23, 233) == 0);
    assert(holo_path_evolve(fixture->object.evolution.path_history,
                            &fixture->terminal, &params) == HOLO_PATH_OK);
    holo_record_evolution(&fixture->object, params.seed, fixture->terminal.steps,
                          fixture->terminal.acc_real, fixture->terminal.acc_imag);
    holo_set_carrier_phase(&fixture->object, 1.0, -1.0);
    assert(holo_verify_software_restoration(&fixture->object, &fixture->initial,
                                            &fixture->terminal, &fixture->restored, 0) == 0);
}

static void fixture_close(Fixture *fixture) {
    fixture_evolve(fixture, 64);
    assert(holo_extract_invariant(&fixture->object) == -1);
    assert(holo_cross_boundary(&fixture->object, fixture->terminal.steps) == 0);
}

static char *read_file(const char *path, long *size_out) {
    FILE *file = fopen(path, "rb");
    long size;
    char *text;
    assert(file != NULL);
    assert(fseek(file, 0, SEEK_END) == 0);
    size = ftell(file);
    assert(size >= 0);
    assert(fseek(file, 0, SEEK_SET) == 0);
    text = (char *)malloc((size_t)size + 1U);
    assert(text != NULL);
    assert(fread(text, 1, (size_t)size, file) == (size_t)size);
    text[size] = '\0';
    fclose(file);
    *size_out = size;
    return text;
}

static void write_file(const char *path, const char *text, size_t size) {
    FILE *file = fopen(path, "wb");
    assert(file != NULL);
    assert(fwrite(text, 1, size, file) == size);
    assert(fclose(file) == 0);
}

static void replace_once(const char *path, const char *from, const char *to) {
    long size;
    char *text = read_file(path, &size);
    char *at = strstr(text, from);
    assert(at != NULL);
    assert(strlen(from) == strlen(to));
    memcpy(at, to, strlen(to));
    write_file(path, text, (size_t)size);
    free(text);
}

static void test_family_and_lifecycle(void) {
    Fixture f;
    size_t i;
    fixture_close(&f);
    assert(f.object.invariant_family.count == HOLO_INVARIANT_COUNT);
    for (i = 0; i < HOLO_INVARIANT_COUNT; ++i) {
        const HoloInvariantRecord *record = &f.object.invariant_family.records[i];
        assert(record->predeclared == 1);
        assert(record->evaluated == 1);
        if (i != HOLO_INV_SERIALIZATION && i != HOLO_INV_SOFTWARE_HOLONOMY) {
            assert(record->passed == 1);
        }
    }
    assert(f.object.invariant_family.records[HOLO_INV_SOFTWARE_HOLONOMY].result ==
           HOLO_INV_RESULT_DEFERRED_NOT_WELL_DEFINED);
    assert(holo_invariant_family_register(&f.object.invariant_family,
                                          HOLO_INV_ORBIT_CONSERVATION) != 0);
    assert(holo_invariant_family_set_tolerance(&f.object.invariant_family,
                                               HOLO_INV_RELATION_BASIS, 1e-6) != 0);
    assert(holo_invariant_family_set_operator(&f.object.invariant_family,
                                              HOLO_INV_RELATION_BASIS, "changed") != 0);
    assert(holo_invariant_family_set_result(&f.object.invariant_family,
                                            HOLO_INV_RELATION_BASIS,
                                            HOLO_INV_RESULT_FAIL) != 0);
    puts("PREDECLARATION_PASS");
    puts("BOUNDARY_GUARD_PASS");
    puts("SEALED_FAMILY_MUTATION_REJECTED_PASS");
    holo_object_destroy(&f.object);
}

static void test_corruption_detection(void) {
    Fixture coordinate, basis, neutral, branch_order, order, parameter;
    HoloPathStep swap;
    uint64_t saved_parameter;
    fixture_evolve(&coordinate, 32);
    coordinate.object.geometry.coordinates[0] += 1.0;
    assert(holo_cross_boundary(&coordinate.object, coordinate.terminal.steps) != 0);
    holo_object_destroy(&coordinate.object);

    fixture_evolve(&basis, 32);
    basis.object.geometry.relation_basis[0] = 1.0;
    assert(holo_cross_boundary(&basis.object, basis.terminal.steps) != 0);
    holo_object_destroy(&basis.object);

    fixture_evolve(&neutral, 32);
    neutral.object.geometry.neutral_reference[0] += 1.0;
    assert(holo_cross_boundary(&neutral.object, neutral.terminal.steps) != 0);
    holo_object_destroy(&neutral.object);

    fixture_evolve(&branch_order, 32);
    branch_order.object.geometry.coordinates[0] = 233.0;
    branch_order.object.geometry.coordinates[1] = 23.0;
    assert(holo_cross_boundary(&branch_order.object, branch_order.terminal.steps) != 0);
    holo_object_destroy(&branch_order.object);

    fixture_evolve(&order, 32);
    swap = order.object.evolution.path_history->steps[0];
    order.object.evolution.path_history->steps[0] = order.object.evolution.path_history->steps[1];
    order.object.evolution.path_history->steps[1] = swap;
    assert(holo_path_history_validate(order.object.evolution.path_history) != HOLO_PATH_OK);
    holo_object_destroy(&order.object);

    fixture_evolve(&parameter, 32);
    saved_parameter = parameter.object.evolution.path_history->steps[3].operator_parameter;
    parameter.object.evolution.path_history->steps[3].operator_parameter ^= UINT64_C(1);
    assert(holo_path_history_validate(parameter.object.evolution.path_history) == HOLO_PATH_ERR_CORRUPT);
    parameter.object.evolution.path_history->steps[3].operator_parameter = saved_parameter;
    holo_object_destroy(&parameter.object);
    puts("ORBIT_COORDINATE_CORRUPTION_DETECTED_PASS");
    puts("RELATION_BASIS_CORRUPTION_DETECTED_PASS");
    puts("NEUTRAL_REFERENCE_CORRUPTION_DETECTED_PASS");
    puts("UNDECLARED_BRANCH_ORDER_CHANGE_DETECTED_PASS");
    puts("PATH_ORDER_CORRUPTION_DETECTED_PASS");
    puts("OPERATOR_PARAMETER_CORRUPTION_DETECTED_PASS");
}

static void test_serialization_recomputation(void) {
    const char *path = "holo_invariant_family_test.holo";
    Fixture f;
    HoloObject loaded;
    HoloObject final_loaded;
    uint64_t original_digest;
    fixture_close(&f);
    original_digest = f.object.invariant_family.family_digest;
    assert(holo_write_json(&f.object, path) == 0);
    assert(holo_read_json(&loaded, path) == 0);
    assert(loaded.invariant_family.family_digest == original_digest);
    assert(loaded.invariant_family.records[HOLO_INV_SERIALIZATION].passed == 1);
    assert(loaded.evolution.operator_seed == 42);
    assert(strcmp(loaded.evolution.continuation_status, "sealed_at_boundary") == 0);
    assert(strcmp(loaded.evolution.closure_status, "software_path_closure_verified") == 0);
    assert(loaded.collapse_boundary.timestamp[0] != '\0');
    assert(holo_write_json(&loaded, path) == 0);
    assert(holo_read_json(&final_loaded, path) == 0);
    assert(holo_invariant_family_equal(&loaded.invariant_family,
                                       &final_loaded.invariant_family, 0));
    printf("original_family_digest=%016llx\n", (unsigned long long)original_digest);
    printf("reloaded_family_digest=%016llx\n",
           (unsigned long long)loaded.invariant_family.family_digest);
    printf("recomputed_family_digest=%016llx\n",
           (unsigned long long)final_loaded.invariant_family.family_digest);
    puts("SERIALIZATION_INVARIANCE_PASS");
    puts("SERIALIZED_VALUES_EQUAL_RECOMPUTED_VALUES_PASS");
    holo_object_destroy(&final_loaded);
    holo_object_destroy(&loaded);
    holo_object_destroy(&f.object);

    fixture_close(&f);
    assert(holo_write_json(&f.object, path) == 0);
    replace_once(path, "\"orbit_sum_initial\":256", "\"orbit_sum_initial\":257");
    assert(holo_read_json(&loaded, path) != 0);
    puts("SERIALIZED_INVARIANT_TAMPERING_DETECTED_PASS");
    assert(holo_write_json(&f.object, path) == 0);
    replace_once(path, "fold_orbit_conservation_v1", "fold_orbit_conservation_v2");
    assert(holo_read_json(&loaded, path) != 0);
    puts("SERIALIZED_OPERATOR_TAMPERING_DETECTED_PASS");
    assert(holo_write_json(&f.object, path) == 0);
    replace_once(path, "orbit_coupled_phase_walk_v1", "orbit_coupled_phase_walk_v2");
    assert(holo_read_json(&loaded, path) != 0);
    puts("SERIALIZED_EVOLUTION_OPERATOR_TAMPERING_DETECTED_PASS");
    assert(holo_write_json(&f.object, path) == 0);
    replace_once(path, "\"terminal_state_digest\": \"f0bdc8e7591e9aab\"",
                       "\"terminal_state_digest\": \"f1bdc8e7591e9aab\"");
    assert(holo_read_json(&loaded, path) != 0);
    puts("SERIALIZED_TERMINAL_DIGEST_TAMPERING_DETECTED_PASS");
    holo_object_destroy(&f.object);
    remove(path);
}

static void print_evidence(void) {
    Fixture f;
    HoloInvariantRecord *orbit;
    HoloInvariantRecord *exchange;
    HoloInvariantRecord *composition;
    fixture_close(&f);
    orbit = &f.object.invariant_family.records[HOLO_INV_ORBIT_CONSERVATION];
    exchange = &f.object.invariant_family.records[HOLO_INV_EXCHANGE_COVARIANCE];
    composition = &f.object.invariant_family.records[HOLO_INV_PATH_COMPOSITION];
    printf("orbit_sums=%.0f,%.0f,%.0f orbit_products=%llu,%llu,%llu result=%s\n",
           orbit->scalar_a, orbit->scalar_b, orbit->scalar_c,
           (unsigned long long)orbit->digest_a, (unsigned long long)orbit->digest_b,
           (unsigned long long)orbit->digest_c, holo_invariant_result_name(orbit->result));
    printf("exchange_neutral_sums=%.0f,%.0f indexed_transformation=%s result=%s\n",
           exchange->scalar_a, exchange->scalar_b,
           exchange->flag_b ? "EXCHANGED" : "FAILED",
           holo_invariant_result_name(exchange->result));
    printf("composition_initial=%016llx terminal=%016llx restored=%016llx identity=%s equality=%s\n",
           (unsigned long long)composition->digest_a, (unsigned long long)composition->digest_b,
           (unsigned long long)composition->digest_c,
           composition->flag_a ? "true" : "false", composition->equality_rule);
    puts("software_path_holonomy=DEFERRED_NOT_WELL_DEFINED");
    puts("missing_structure=group_valued_carrier_transform_per_path_step");
    holo_object_destroy(&f.object);
}

int main(void) {
    test_family_and_lifecycle();
    test_corruption_detection();
    test_serialization_recomputation();
    print_evidence();
    puts("CORRUPTION_MATRIX: fold_coordinate=orbit_conservation/reconstruction");
    puts("CORRUPTION_MATRIX: relation_basis=relation_basis");
    puts("CORRUPTION_MATRIX: neutral_reference=relation_basis");
    puts("CORRUPTION_MATRIX: undeclared_branch_order=path_state_digest");
    puts("CORRUPTION_MATRIX: path_step_swap=path_order/continuity");
    puts("CORRUPTION_MATRIX: operator_parameter=step_digest");
    puts("CORRUPTION_MATRIX: terminal_digest=path_continuity");
    puts("CORRUPTION_MATRIX: serialized_result=family_digest/recomputation");
    puts("CORRUPTION_MATRIX: post_boundary_add=lifecycle_guard");
    puts("HOLO_INVARIANT_FAMILY_TEST_PASS");
    return 0;
}
