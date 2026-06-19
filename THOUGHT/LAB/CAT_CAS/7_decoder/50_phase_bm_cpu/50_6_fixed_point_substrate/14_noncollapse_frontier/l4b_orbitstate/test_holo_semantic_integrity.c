#include "orbit_state.h"
#include "holo_path_history.h"
#include "../holo_runtime/holo_geometry.h"
#include "../holo_runtime/holo_semantic_integrity.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv_byte(uint64_t digest, unsigned char byte) {
    return (digest ^ (uint64_t)byte) * UINT64_C(1099511628211);
}

static uint64_t fnv_u64(uint64_t digest, uint64_t value) {
    unsigned int shift;
    for (shift = 0; shift < 64U; shift += 8U) {
        digest = fnv_byte(digest, (unsigned char)(value >> shift));
    }
    return digest;
}

static uint64_t fnv_text(uint64_t digest, const char *text, size_t limit) {
    size_t i;
    for (i = 0; i < limit && text[i] != '\0'; ++i) {
        digest = fnv_byte(digest, (unsigned char)text[i]);
    }
    return digest;
}

static uint64_t recompute_step_digest(const HoloPathStep *step) {
    uint64_t digest = UINT64_C(14695981039346656037);
    digest = fnv_u64(digest, step->step_index);
    digest = fnv_text(digest, step->operator_id, sizeof(step->operator_id));
    digest = fnv_u64(digest, step->operator_parameter);
    digest = fnv_u64(digest, step->pre_acc_real_bits);
    digest = fnv_u64(digest, step->pre_acc_imag_bits);
    digest = fnv_u64(digest, step->post_acc_real_bits);
    digest = fnv_u64(digest, step->post_acc_imag_bits);
    digest = fnv_u64(digest, step->pre_state_digest);
    digest = fnv_u64(digest, step->post_state_digest);
    return digest;
}

static char *read_all(const char *path, size_t *size_out) {
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
    assert(fclose(file) == 0);
    *size_out = (size_t)size;
    return text;
}

static void replace_once(const char *path, const char *from, const char *to) {
    size_t size;
    char *text = read_all(path, &size);
    char *at = strstr(text, from);
    FILE *file;
    assert(at != NULL);
    assert(strlen(from) == strlen(to));
    memcpy(at, to, strlen(to));
    file = fopen(path, "wb");
    assert(file != NULL);
    assert(fwrite(text, 1, size, file) == size);
    assert(fclose(file) == 0);
    free(text);
}

static void make_open_object(HoloObject *object, OrbitState *initial,
                             OrbitState *terminal) {
    EvolParams params = { .max_steps = 16, .seed = 42 };
    OrbitState restored;
    orbit_init(initial, 256, 23, 233);
    *terminal = *initial;
    assert(holo_object_init(object, 42, 256, 23, 233) == 0);
    assert(holo_path_evolve(object->evolution.path_history, terminal, &params) == HOLO_PATH_OK);
    holo_record_evolution(object, params.seed, terminal->steps,
                          terminal->acc_real, terminal->acc_imag);
    assert(holo_verify_software_restoration(object, initial, terminal, &restored, 0) == 0);
    assert(holo_orbit_state_equal_bitwise(initial, &restored));
}

static void test_semantic_forgery_detection(void) {
    HoloObject object;
    OrbitState initial;
    OrbitState terminal;
    HoloPathStep *step;
    uint64_t original_parameter;
    uint64_t original_digest;

    make_open_object(&object, &initial, &terminal);
    assert(holo_object_validate_semantic(&object));

    step = &object.evolution.path_history->steps[3];
    original_parameter = step->operator_parameter;
    original_digest = step->step_digest;
    step->operator_parameter = (uint64_t)object.N + 1U;
    step->step_digest = recompute_step_digest(step);

    assert(holo_path_history_validate(object.evolution.path_history) == HOLO_PATH_OK);
    assert(holo_path_history_validate_semantic(object.evolution.path_history,
                                               &initial, NULL) == HOLO_PATH_ERR_OPERATOR);
    puts("SELF_CONSISTENT_PARAMETER_FORGERY_REJECTED_PASS");

    step->operator_parameter = original_parameter;
    step->step_digest = original_digest;
    assert(holo_object_validate_semantic(&object));
    holo_object_destroy(&object);
}

static void test_atomic_boundary_rollback(void) {
    HoloObject object;
    OrbitState initial;
    OrbitState terminal;
    double original_basis;

    make_open_object(&object, &initial, &terminal);
    original_basis = object.geometry.relation_basis[0];
    object.geometry.relation_basis[0] = 1.0;
    assert(holo_cross_boundary_atomic(&object, terminal.steps) != 0);
    assert(!object.collapse_boundary.crossed);
    assert(!object.evolution.path_history->sealed);
    assert(object.evolution.path_history->appendable);
    assert(!object.invariant_family.extracted);
    puts("FAILED_BOUNDARY_TRANSITION_ROLLED_BACK_PASS");

    object.geometry.relation_basis[0] = original_basis;
    assert(holo_cross_boundary_atomic(&object, terminal.steps) == 0);
    assert(holo_object_validate_semantic(&object));
    puts("ATOMIC_BOUNDARY_COMMIT_PASS");
    holo_object_destroy(&object);
}

static void test_strict_reader(void) {
    const char *path = "holo_semantic_integrity_test.holo";
    HoloObject object;
    HoloObject loaded;
    OrbitState initial;
    OrbitState terminal;
    OrbitState restored;

    make_open_object(&object, &initial, &terminal);
    assert(holo_cross_boundary_atomic(&object, terminal.steps) == 0);
    assert(holo_write_json(&object, path) == 0);
    assert(holo_read_json_strict(&loaded, path) == 0);
    assert(holo_verify_software_restoration(&loaded, &initial, &terminal,
                                             &restored, 1) == 0);
    assert(holo_object_validate_semantic(&loaded));
    holo_object_destroy(&loaded);
    puts("STRICT_READER_VALID_ARTIFACT_PASS");

    replace_once(path, "\"crossed\": true", "\"crossed\": null");
    assert(holo_read_json_strict(&loaded, path) != 0);
    puts("SERIALIZED_BOUNDARY_TAMPERING_REJECTED_PASS");

    assert(holo_write_json(&object, path) == 0);
    replace_once(path, "\"restored\": true", "\"restored\": null");
    assert(holo_read_json_strict(&loaded, path) != 0);
    puts("SERIALIZED_RESTORATION_TAMPERING_REJECTED_PASS");

    remove(path);
    holo_object_destroy(&object);
}

int main(void) {
    test_semantic_forgery_detection();
    test_atomic_boundary_rollback();
    test_strict_reader();
    puts("HOLO_SEMANTIC_INTEGRITY_TEST_PASS");
    return 0;
}
