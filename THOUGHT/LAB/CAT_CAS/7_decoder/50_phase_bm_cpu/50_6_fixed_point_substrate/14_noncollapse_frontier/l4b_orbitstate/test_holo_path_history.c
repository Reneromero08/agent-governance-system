#include "holo_path_history.h"
#include "../holo_runtime/holo_geometry.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int file_contains(const char *path, const char *needle) {
    FILE *file = fopen(path, "rb");
    long size;
    char *text;
    int found;
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
    found = strstr(text, needle) != NULL;
    free(text);
    return found;
}

static void test_append_order_and_growth(void) {
    OrbitState initial;
    OrbitState state;
    HoloPathHistory history;
    HoloPathStep invalid;
    orbit_init(&initial, 256, 23, 233);
    state = initial;
    assert(holo_path_history_init(&history, 1U, &initial) == HOLO_PATH_OK);
    assert(holo_path_apply_step(&history, &state, 3U) == HOLO_PATH_OK);
    assert(history.count == 1U);
    invalid = history.steps[0];
    invalid.step_index = 2U;
    assert(holo_path_history_append(&history, &invalid) == HOLO_PATH_ERR_ORDER);
    assert(history.count == 1U);
    assert(holo_path_apply_step(&history, &state, 7U) == HOLO_PATH_OK);
    assert(history.count == 2U);
    assert(history.capacity >= 2U);
    assert(history.steps[0].post_state_digest == history.steps[1].pre_state_digest);
    assert(holo_path_history_validate(&history) == HOLO_PATH_OK);
    holo_path_history_destroy(&history);
    puts("APPEND_ORDER_PASS");
    puts("INVALID_ORDER_REJECTED_PASS");
    puts("CAPACITY_GROWTH_PASS");
}

static void test_reverse_corruption_and_seal(void) {
    OrbitState initial;
    OrbitState terminal;
    OrbitState restored;
    OrbitState unchanged;
    EvolParams params = { .max_steps = 128, .seed = 42 };
    HoloPathHistory history;
    HoloPathStep swap;
    uint64_t parameter;
    orbit_init(&initial, 256, 23, 233);
    terminal = initial;
    assert(holo_path_history_init(&history, 2U, &initial) == HOLO_PATH_OK);
    assert(holo_path_evolve(&history, &terminal, &params) == HOLO_PATH_OK);
    assert(holo_path_reverse(&history, &terminal, &restored) == HOLO_PATH_OK);
    assert(holo_orbit_state_equal_bitwise(&initial, &restored));

    swap = history.steps[0];
    history.steps[0] = history.steps[1];
    history.steps[1] = swap;
    assert(holo_path_history_validate(&history) != HOLO_PATH_OK);
    swap = history.steps[0];
    history.steps[0] = history.steps[1];
    history.steps[1] = swap;
    assert(holo_path_history_validate(&history) == HOLO_PATH_OK);

    parameter = history.steps[3].operator_parameter;
    history.steps[3].operator_parameter ^= UINT64_C(1);
    assert(holo_path_history_validate(&history) == HOLO_PATH_ERR_CORRUPT);
    history.steps[3].operator_parameter = parameter;
    assert(holo_path_history_validate(&history) == HOLO_PATH_OK);

    assert(holo_path_history_seal(&history) == HOLO_PATH_OK);
    unchanged = terminal;
    assert(holo_path_apply_step(&history, &terminal, 9U) == HOLO_PATH_ERR_SEALED);
    assert(holo_orbit_state_equal_bitwise(&unchanged, &terminal));
    holo_path_history_destroy(&history);
    puts("FORWARD_REVERSE_BITWISE_PASS");
    puts("WRONG_ORDER_CORRUPTION_REJECTED_PASS");
    puts("MUTATED_STEP_REJECTED_PASS");
    puts("SEALED_APPEND_REJECTED_PASS");
}

static void test_serialized_roundtrip(void) {
    const char *path = "holo_path_history_test.holo";
    OrbitState initial;
    OrbitState terminal;
    OrbitState restored;
    EvolParams params = { .max_steps = 512, .seed = 42 };
    HoloObject object;
    HoloPathHistory *reloaded = NULL;
    uint64_t saved_parameter;

    orbit_init(&initial, 256, 23, 233);
    terminal = initial;
    assert(holo_object_init(&object, 42, 256, 23, 233) == 0);
    assert(holo_path_evolve(object.evolution.path_history, &terminal, &params) == HOLO_PATH_OK);
    holo_record_evolution(&object, params.seed, terminal.steps,
                          terminal.acc_real, terminal.acc_imag);
    holo_set_carrier_phase(&object, 1.0, -1.0);
    assert(holo_extract_invariant(&object) == -1);
    assert(holo_verify_software_restoration(&object, &initial, &terminal,
                                            &restored, 0) == 0);
    assert(holo_cross_boundary(&object, terminal.steps) == 0);
    assert(holo_write_json(&object, path) == 0);

    holo_path_history_destroy(object.evolution.path_history);
    free(object.evolution.path_history);
    object.evolution.path_history = NULL;
    assert(holo_path_history_read_file(path, &reloaded) == HOLO_PATH_OK);
    assert(reloaded->count == 512U);
    assert(holo_replace_path_history(&object, reloaded) == 0);
    assert(holo_verify_software_restoration(&object, &initial, &terminal,
                                            &restored, 1) == 0);
    assert(holo_orbit_state_equal_bitwise(&initial, &restored));
    assert(object.evolution.path_history->serialized_roundtrip == 1);
    assert(holo_write_json(&object, path) == 0);

    saved_parameter = object.evolution.path_history->steps[17].operator_parameter;
    object.evolution.path_history->steps[17].operator_parameter ^= UINT64_C(1);
    assert(holo_path_history_validate(object.evolution.path_history) == HOLO_PATH_ERR_CORRUPT);
    object.evolution.path_history->steps[17].operator_parameter = saved_parameter;
    assert(holo_path_history_validate(object.evolution.path_history) == HOLO_PATH_OK);

    assert(file_contains(path, "\"path_history\""));
    assert(file_contains(path, "\"count\": 512"));
    assert(file_contains(path, "\"serialized_roundtrip\": true"));
    assert(file_contains(path, "\"restoration_verified\": true"));
    assert(!file_contains(path, "\"winner\""));
    assert(!file_contains(path, "\"candidate_score\""));
    assert(!file_contains(path, "\"hidden_d\""));
    assert(!file_contains(path, "\"recovered_d\""));
    assert(!file_contains(path, "\"orientation_label\""));
    assert(!file_contains(path, "\"verify_pass\""));
    assert(!file_contains(path, "\"AUC\""));

    puts("SERIALIZED_HISTORY_ROUNDTRIP_PASS");
    puts("PRE_BOUNDARY_INVARIANT_GUARD_PASS");
    puts("FORBIDDEN_SERIALIZED_FIELDS_PASS");
    puts("PATH REVERSIBILITY TEST");
    puts("PATH_REVERSIBILITY_PASS");
    printf("initial_state={N:%d,lower:%d,mirror:%d,acc_real:%.17g,acc_imag:%.17g,steps:%d}\n",
           initial.N, initial.branch_plus, initial.branch_minus,
           initial.acc_real, initial.acc_imag, initial.steps);
    printf("terminal_state={N:%d,lower:%d,mirror:%d,acc_real:%.17g,acc_imag:%.17g,steps:%d}\n",
           terminal.N, terminal.branch_plus, terminal.branch_minus,
           terminal.acc_real, terminal.acc_imag, terminal.steps);
    printf("restored_state={N:%d,lower:%d,mirror:%d,acc_real:%.17g,acc_imag:%.17g,steps:%d}\n",
           restored.N, restored.branch_plus, restored.branch_minus,
           restored.acc_real, restored.acc_imag, restored.steps);
    printf("initial_digest=%016llx\n",
           (unsigned long long)object.evolution.path_history->initial_state_digest);
    printf("terminal_digest=%016llx\n",
           (unsigned long long)object.evolution.path_history->terminal_state_digest);
    printf("restored_digest=%016llx\n",
           (unsigned long long)object.evolution.path_history->restored_state_digest);
    printf("steps=%zu\n", object.evolution.path_history->count);
    printf("serialized_history=%s\n", path);
    printf("reloaded_history_count=%zu\n", object.evolution.path_history->count);
    puts("equality=bitwise_numeric_orbit_state");
    puts("serialized_roundtrip=true");

    holo_object_destroy(&object);
    remove(path);
}

int main(void) {
    test_append_order_and_growth();
    test_reverse_corruption_and_seal();
    test_serialized_roundtrip();
    puts("HOLO_PATH_HISTORY_TEST_PASS");
    return 0;
}
